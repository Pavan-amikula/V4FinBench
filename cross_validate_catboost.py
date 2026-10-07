"""Five-fold CatBoost evaluation for V4FinBench company_years_h2.

The folds and row order are the same as the Logistic Regression, LightGBM, and
XGBoost experiments. For each rotation, the validation fold selects one model
configuration and its F1 threshold. The next fold is then used for testing.

The grid follows the published V4FinBench CatBoost grid:
iterations={100, 200}, depth={4, 6, 8}, learning_rate={0.01, 0.05, 0.1}.
All prepared features are numeric codes/ratios, so this comparison deliberately
uses CatBoost as a numeric tabular model, without class weights or categorical
feature declarations.
"""

from pathlib import Path
from datetime import datetime, timezone
import gc
import hashlib
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

CONFIGURATIONS = [
    {
        "iterations": iterations,
        "depth": depth,
        "learning_rate": learning_rate,
    }
    for iterations in [100, 200]
    for depth in [4, 6, 8]
    for learning_rate in [0.01, 0.05, 0.1]
]


def validate_inputs(metadata, folds, feature_columns):
    if folds.ndim != 1 or len(folds) != len(metadata):
        raise ValueError("Fold assignments must match dataset rows exactly.")
    if set(np.unique(folds)) != set(range(NUMBER_OF_FOLDS)):
        raise ValueError("Expected exactly fold values 0, 1, 2, 3, and 4.")
    required = ["country", "company", TARGET_COLUMN]
    if metadata[required].isna().any().any():
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
    for fold_number in range(NUMBER_OF_FOLDS):
        if metadata.loc[folds == fold_number, TARGET_COLUMN].nunique() != 2:
            raise ValueError(f"Fold {fold_number} must contain both target classes.")


def select_threshold(y_true, scores):
    if not np.isfinite(scores).all():
        raise ValueError("CatBoost produced non-finite scores.")
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5
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
    import joblib
    import pyarrow.parquet as pq
    import sklearn
    import catboost
    from catboost import CatBoostClassifier

    started = time.perf_counter()
    for path in [DATA_PATH, FEATURE_PATH, FOLD_PATH]:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    feature_columns = [
        line.strip()
        for line in FEATURE_PATH.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Feature names must be nonempty and unique.")

    print("STEP 1: Checking saved folds", flush=True)
    metadata = pq.read_table(
        DATA_PATH,
        columns=["country", "company", TARGET_COLUMN],
    ).to_pandas().reset_index(drop=True)
    folds = np.loadtxt(FOLD_PATH, dtype=np.int8)
    validate_inputs(metadata, folds, feature_columns)
    labels = metadata[TARGET_COLUMN].to_numpy(dtype=np.int8)
    print(pd.crosstab(folds, labels).to_string(), flush=True)
    print("Company-country grouping check passed.", flush=True)
    del metadata
    gc.collect()

    print("\nSTEP 2: Loading feature matrix", flush=True)
    frame = pq.read_table(DATA_PATH, columns=feature_columns).to_pandas()
    X_all = frame.to_numpy(dtype=np.float32, na_value=np.nan, copy=True)
    del frame
    for column_number in range(X_all.shape[1]):
        infinite = np.isinf(X_all[:, column_number])
        if infinite.any():
            X_all[infinite, column_number] = np.nan
    gc.collect()
    print(f"Rows: {len(X_all):,}; features: {len(feature_columns)}", flush=True)

    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = ROOT / "results" / f"h1_catboost_cv_{run_name}"
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "dataset": DATA_PATH.name,
                "paper_horizon": 1,
                "feature_count": len(feature_columns),
                "feature_columns": feature_columns,
                "fold_file_sha256": hashlib.sha256(FOLD_PATH.read_bytes()).hexdigest(),
                "catboost_version": catboost.__version__,
                "sklearn_version": sklearn.__version__,
                "configurations": CONFIGURATIONS,
                "selection": "highest validation average precision; validation F1 threshold",
                "class_weight": "none",
                "categorical_features": "none; all prepared columns supplied as numeric",
                "evaluation_note": "Exploratory grouped CV; operational-status timing and official-fold equivalence are not independently verified.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fold_metrics = []
    all_test_predictions = []
    test_coverage = np.zeros(len(X_all), dtype=np.int8)

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

        X_train, y_train = X_all[train_indices], labels[train_indices]
        X_validation, y_validation = X_all[validation_indices], labels[validation_indices]
        X_test, y_test = X_all[test_indices], labels[test_indices]
        destination = output_dir / f"validation_{validation_fold}_test_{test_fold}"
        destination.mkdir()
        print(
            f"\nROTATION {validation_fold + 1}/{NUMBER_OF_FOLDS}: "
            f"train={train_folds}, validation={validation_fold}, test={test_fold}",
            flush=True,
        )

        search_rows = []
        best_model = None
        best_configuration = None
        best_validation_ap = -1.0

        for candidate_number, configuration in enumerate(CONFIGURATIONS, start=1):
            print(
                f"  [{candidate_number}/{len(CONFIGURATIONS)}] "
                f"iterations={configuration['iterations']}, "
                f"depth={configuration['depth']}, "
                f"learning_rate={configuration['learning_rate']}",
                flush=True,
            )
            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="Logloss",
                random_seed=42,
                thread_count=-1,
                verbose=False,
                allow_writing_files=False,
                l2_leaf_reg=3.0,
                **configuration,
            )
            model.fit(X_train, y_train, verbose=False)
            validation_scores = model.predict_proba(X_validation)[:, 1]
            threshold = select_threshold(y_validation, validation_scores)
            validation_metrics = calculate_metrics(
                y_validation, validation_scores, threshold
            )
            search_row = {
                **configuration,
                "threshold": threshold,
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
            search_rows.append(search_row)
            pd.DataFrame(search_rows).sort_values(
                "validation_average_precision", ascending=False
            ).to_csv(destination / "configuration_search.csv", index=False)
            print(
                f"    validation AP={validation_metrics['average_precision']:.6f}; "
                f"F1={validation_metrics['f1']:.6f}; "
                f"threshold={threshold:.6f}",
                flush=True,
            )
            if validation_metrics["average_precision"] > best_validation_ap:
                best_validation_ap = validation_metrics["average_precision"]
                best_model = model
                best_configuration = search_row
            else:
                del model
            gc.collect()

        print(
            "  SELECTED: "
            f"iterations={best_configuration['iterations']}, "
            f"depth={best_configuration['depth']}, "
            f"learning_rate={best_configuration['learning_rate']}; "
            f"validation AP={best_validation_ap:.6f}",
            flush=True,
        )

        test_scores = best_model.predict_proba(X_test)[:, 1]
        test_metrics = calculate_metrics(
            y_test, test_scores, best_configuration["threshold"]
        )
        test_coverage[test_indices] += 1
        all_test_predictions.append(
            pd.DataFrame(
                {
                    "row_index": test_indices,
                    "actual_label": y_test,
                    "predicted_probability": test_scores,
                    "predicted_label": (
                        test_scores >= best_configuration["threshold"]
                    ).astype(np.int8),
                    "validation_fold": validation_fold,
                    "test_fold": test_fold,
                }
            )
        )
        fold_metrics.append(
            {
                "validation_fold": validation_fold,
                "test_fold": test_fold,
                "train_folds": ",".join(map(str, train_folds)),
                "selected_iterations": best_configuration["iterations"],
                "selected_depth": best_configuration["depth"],
                "selected_learning_rate": best_configuration["learning_rate"],
                "threshold": best_configuration["threshold"],
                "validation_average_precision": best_validation_ap,
                "validation_f1": best_configuration["validation_f1"],
                "test_rows": len(test_indices),
                "test_positives": int(y_test.sum()),
                **test_metrics,
                "rotation_seconds": time.perf_counter() - rotation_started,
            }
        )

        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": best_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False).to_csv(
            destination / "feature_importance.csv", index=False
        )
        best_model.save_model(str(destination / "catboost_model.cbm"))
        joblib.dump(
            {
                "feature_columns": feature_columns,
                "model": best_model,
                "threshold": best_configuration["threshold"],
                "score_type": "probability",
                "configuration": best_configuration,
            },
            destination / "model.joblib",
        )
        pd.DataFrame(fold_metrics).to_csv(output_dir / "fold_metrics.csv", index=False)
        print(
            f"  TEST: AP={test_metrics['average_precision']:.6f}; "
            f"F1={test_metrics['f1']:.6f}; "
            f"precision={test_metrics['precision']:.6f}; "
            f"recall={test_metrics['recall']:.6f}",
            flush=True,
        )

        del X_train, y_train, X_validation, y_validation, X_test, y_test
        del best_model, best_configuration, test_scores
        gc.collect()

    if not np.all(test_coverage == 1):
        raise RuntimeError("Each row must be used as test data exactly once.")

    results = pd.DataFrame(fold_metrics)
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]
    summary = (
        results[metric_names]
        .agg(["mean", "std"])
        .T.rename_axis("metric")
        .reset_index()
    )
    summary.to_csv(output_dir / "cv_summary.csv", index=False)
    pd.concat(all_test_predictions, ignore_index=True).sort_values(
        "row_index"
    ).to_parquet(output_dir / "all_test_predictions.parquet", index=False)

    print("\nFIVE-FOLD CATBOOST RESULTS (std = sample standard deviation)", flush=True)
    print(
        results[
            [
                "validation_fold",
                "test_fold",
                "selected_iterations",
                "selected_depth",
                "selected_learning_rate",
                *metric_names,
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nMEAN AND STANDARD DEVIATION", flush=True)
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        ),
        flush=True,
    )
    print(f"\nTotal runtime: {time.perf_counter() - started:.2f} seconds", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
