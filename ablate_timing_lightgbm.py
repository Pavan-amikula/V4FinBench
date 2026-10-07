"""Feature-timing sensitivity check for the V4FinBench h=1 task.

This is not a proof that a feature is leaked. It measures how much the
current LightGBM baseline changes when potentially timing-sensitive metadata
(`operational_status` and `year`) is removed. Every variant uses the same
company-grouped folds, the same tree settings, validation-only threshold
selection, and the next fold as test data.
"""

from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/raw/company_years_h2.parquet"
FEATURE_PATH = ROOT / "data/processed/feature_columns.txt"
FOLD_PATH = ROOT / "data/folds/h2/fold_assignments.txt"
TARGET_COLUMN = "main_label"
NUMBER_OF_FOLDS = 5

VARIANT_REMOVALS = {
    "all_features": [],
    "without_operational_status": ["operational_status"],
    "without_year": ["year"],
    "without_status_and_year": ["operational_status", "year"],
}


def validate_inputs(metadata, folds, feature_columns):
    if folds.ndim != 1 or len(folds) != len(metadata):
        raise ValueError("Fold assignments must match dataset rows exactly.")
    if set(np.unique(folds)) != set(range(NUMBER_OF_FOLDS)):
        raise ValueError("Expected fold values 0, 1, 2, 3, and 4.")
    if metadata[["country", "company", TARGET_COLUMN]].isna().any().any():
        raise ValueError("Country, company, and target cannot contain missing values.")
    if not metadata[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("main_label must contain only 0 and 1.")
    forbidden = {TARGET_COLUMN, "company", "emis_id", "link", "num"}
    leaked = forbidden.intersection(feature_columns)
    if leaked:
        raise ValueError(f"Target or identifier columns in features: {sorted(leaked)}")
    grouped = metadata[["country", "company"]].copy()
    grouped["fold"] = folds
    group_folds = grouped.groupby(
        ["country", "company"], observed=True
    )["fold"].nunique()
    if (group_folds > 1).any():
        raise ValueError("A company-country group occurs in more than one fold.")


def select_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.0
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return float(thresholds[int(np.argmax(f1_values))])


def calculate_metrics(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    import lightgbm as lgb
    import pyarrow.parquet as pq
    import sklearn

    started = time.perf_counter()
    for path in [DATA_PATH, FEATURE_PATH, FOLD_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    features = [
        line.strip()
        for line in FEATURE_PATH.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    metadata = pq.read_table(
        DATA_PATH,
        columns=["country", "company", TARGET_COLUMN],
    ).to_pandas().reset_index(drop=True)
    folds = np.loadtxt(FOLD_PATH, dtype=np.int8)
    validate_inputs(metadata, folds, features)
    labels = metadata[TARGET_COLUMN].to_numpy(dtype=np.int8)
    print("STEP 1: Fold and grouping checks passed", flush=True)
    print(pd.crosstab(folds, labels).to_string(), flush=True)
    del metadata
    gc.collect()

    print("\nSTEP 2: Loading feature matrix", flush=True)
    frame = pq.read_table(DATA_PATH, columns=features).to_pandas()
    X_all = frame.to_numpy(dtype=np.float32, na_value=np.nan, copy=True)
    del frame
    for column_number in range(X_all.shape[1]):
        infinite = np.isinf(X_all[:, column_number])
        if infinite.any():
            X_all[infinite, column_number] = np.nan
    gc.collect()
    print(f"Rows: {len(X_all):,}; features: {len(features)}", flush=True)

    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = ROOT / "results" / f"h1_lightgbm_timing_ablation_{run_name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "dataset": DATA_PATH.name,
                "paper_horizon": 1,
                "feature_count": len(features),
                "feature_file_sha256": hashlib.sha256(
                    FEATURE_PATH.read_bytes()
                ).hexdigest(),
                "fold_file_sha256": hashlib.sha256(
                    FOLD_PATH.read_bytes()
                ).hexdigest(),
                "lightgbm_version": lgb.__version__,
                "sklearn_version": sklearn.__version__,
                "variants": VARIANT_REMOVALS,
                "fixed_model": {
                    "n_estimators": 1500,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "min_child_samples": 100,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "scale_pos_weight": 1.0,
                },
                "selection": "early stopping on validation average precision; threshold maximizes validation F1",
                "interpretation": "Removing a feature tests sensitivity, not leakage proof; point-in-time availability requires domain evidence.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    variant_rows = []

    for variant_name, removals in VARIANT_REMOVALS.items():
        variant_features = [feature for feature in features if feature not in removals]
        feature_indices = np.array(
            [features.index(feature) for feature in variant_features],
            dtype=np.int64,
        )
        X_variant = X_all[:, feature_indices]
        variant_dir = output_dir / variant_name
        variant_dir.mkdir()
        print(
            f"\nVARIANT: {variant_name} ({len(variant_features)} features)",
            flush=True,
        )

        for validation_fold in range(NUMBER_OF_FOLDS):
            rotation_started = time.perf_counter()
            test_fold = (validation_fold + 1) % NUMBER_OF_FOLDS
            train_folds = [
                fold for fold in range(NUMBER_OF_FOLDS)
                if fold not in [validation_fold, test_fold]
            ]
            train_indices = np.flatnonzero(np.isin(folds, train_folds))
            validation_indices = np.flatnonzero(folds == validation_fold)
            test_indices = np.flatnonzero(folds == test_fold)
            X_train = X_variant[train_indices]
            y_train = labels[train_indices]
            X_validation = X_variant[validation_indices]
            y_validation = labels[validation_indices]
            X_test = X_variant[test_indices]
            y_test = labels[test_indices]

            model = lgb.LGBMClassifier(
                objective="binary",
                boosting_type="gbdt",
                metric="average_precision",
                n_estimators=1500,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=100,
                subsample=0.80,
                subsample_freq=1,
                colsample_bytree=0.80,
                reg_alpha=0.0,
                reg_lambda=1.0,
                max_bin=255,
                scale_pos_weight=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
                deterministic=True,
                force_col_wise=True,
            )
            fit_args = {
                "eval_names": ["validation"],
                "eval_metric": "average_precision",
                "feature_name": variant_features,
                "callbacks": [
                    lgb.early_stopping(
                        stopping_rounds=100,
                        first_metric_only=True,
                        verbose=False,
                    )
                ],
            }
            if "eval_X" in inspect.signature(model.fit).parameters:
                fit_args.update(eval_X=X_validation, eval_y=y_validation)
            else:
                fit_args["eval_set"] = [(X_validation, y_validation)]
            model.fit(X_train, y_train, **fit_args)
            best_iteration = int(model.best_iteration_)
            validation_scores = model.booster_.predict(
                X_validation,
                raw_score=True,
                num_iteration=best_iteration,
            )
            threshold = select_threshold(y_validation, validation_scores)
            validation_metrics = calculate_metrics(
                y_validation, validation_scores, threshold
            )
            test_scores = model.booster_.predict(
                X_test,
                raw_score=True,
                num_iteration=best_iteration,
            )
            test_metrics = calculate_metrics(y_test, test_scores, threshold)

            row = {
                "variant": variant_name,
                "removed_features": ",".join(removals),
                "feature_count": len(variant_features),
                "validation_fold": validation_fold,
                "test_fold": test_fold,
                "train_folds": ",".join(map(str, train_folds)),
                "best_iteration": best_iteration,
                "threshold": threshold,
                "validation_average_precision": validation_metrics[
                    "average_precision"
                ],
                "validation_f1": validation_metrics["f1"],
                "test_rows": len(test_indices),
                "test_positives": int(y_test.sum()),
                **test_metrics,
                "rotation_seconds": time.perf_counter() - rotation_started,
            }
            variant_rows.append(row)
            pd.DataFrame(variant_rows).to_csv(
                output_dir / "ablation_fold_metrics.csv", index=False
            )
            pd.DataFrame(
                {
                    "row_index": test_indices,
                    "actual_label": y_test,
                    "decision_score": test_scores,
                    "predicted_label": (
                        test_scores >= threshold
                    ).astype(np.int8),
                    "validation_fold": validation_fold,
                    "test_fold": test_fold,
                }
            ).to_parquet(
                variant_dir / f"test_predictions_v{validation_fold}_t{test_fold}.parquet",
                index=False,
            )
            pd.DataFrame(
                {
                    "feature": variant_features,
                    "gain_importance": model.booster_.feature_importance(
                        importance_type="gain"
                    ),
                }
            ).sort_values("gain_importance", ascending=False).to_csv(
                variant_dir / f"feature_importance_v{validation_fold}_t{test_fold}.csv",
                index=False,
            )
            print(
                f"  rotation {validation_fold + 1}/5: trees={best_iteration}; "
                f"test AP={test_metrics['average_precision']:.6f}; "
                f"test F1={test_metrics['f1']:.6f}",
                flush=True,
            )
            del model, X_train, y_train, X_validation, y_validation, X_test, y_test
            del validation_scores, test_scores
            gc.collect()
        del X_variant
        gc.collect()

    results = pd.DataFrame(variant_rows)
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]
    summary = (
        results.groupby("variant", as_index=False)[metric_names]
        .agg(["mean", "std"])
    )
    summary.to_csv(output_dir / "ablation_summary.csv")
    print("\nABLATION SUMMARY (mean ± sample standard deviation)", flush=True)
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"), flush=True)
    print(f"\nTotal runtime: {time.perf_counter() - started:.2f} seconds", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
