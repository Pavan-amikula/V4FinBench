from pathlib import Path
from datetime import datetime, timezone
import gc
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from lightgbm import LGBMClassifier
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


DATA_PATH = Path("data/raw/company_years_h2.parquet")
FEATURE_PATH = Path("data/processed/feature_columns.txt")

timestamp = datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%S_%fZ"
)

OUTPUT_DIR = Path(
    f"results/h1_lightgbm_temporal_{timestamp}"
)

TARGET_COLUMN = "main_label"
TRAIN_END_YEAR = 2017
VALIDATION_YEAR = 2018
TEST_YEAR = 2019

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

start_time = time.perf_counter()


def find_best_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        scores
    )

    if len(thresholds) == 0:
        return 0.0

    precision = precision[:-1]
    recall = recall[:-1]

    f1_values = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(recall),
        where=(precision + recall) != 0
    )

    return float(thresholds[int(np.argmax(f1_values))])


def evaluate(y_true, scores, threshold):
    predictions = (
        scores >= threshold
    ).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_true,
            scores
        ),
        "average_precision": average_precision_score(
            y_true,
            scores
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "predictions": predictions,
    }


print("STEP 1: Loading feature names")

feature_columns = [
    line.strip()
    for line in FEATURE_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

if "year" not in feature_columns:
    raise ValueError("The year feature was not found.")

if "operational_status" not in feature_columns:
    raise ValueError(
        "The operational_status feature was not found."
    )

print(f"Features: {len(feature_columns)}")


print("\nSTEP 2: Loading dataset")

data = pq.read_table(
    DATA_PATH,
    columns=feature_columns + [TARGET_COLUMN]
).to_pandas()

years = data["year"].to_numpy(
    dtype=np.int16,
    copy=True
)

y_all = data.pop(TARGET_COLUMN).to_numpy(
    dtype=np.int8
)

X_all = data.to_numpy(
    dtype=np.float32,
    na_value=np.nan,
    copy=True
)

del data
gc.collect()

infinite_count = int(np.isinf(X_all).sum())
X_all[np.isinf(X_all)] = np.nan

print(f"Rows: {len(y_all):,}")
print(f"Infinite values changed: {infinite_count:,}")


print("\nSTEP 3: Creating temporal split")

train_indices = np.flatnonzero(
    years <= TRAIN_END_YEAR
)

validation_indices = np.flatnonzero(
    years == VALIDATION_YEAR
)

test_indices = np.flatnonzero(
    years == TEST_YEAR
)

split_rows = []

for split_name, indices in [
    ("training", train_indices),
    ("validation", validation_indices),
    ("testing", test_indices),
]:
    positives = int(y_all[indices].sum())

    split_rows.append({
        "split": split_name,
        "rows": len(indices),
        "positive_rows": positives,
        "positive_percentage": (
            positives / len(indices) * 100
        ),
    })

split_summary = pd.DataFrame(split_rows)

print(split_summary.to_string(index=False))

split_summary.to_csv(
    OUTPUT_DIR / "split_summary.csv",
    index=False
)


variants = {
    "all_features": feature_columns,
    "without_operational_status": [
        feature
        for feature in feature_columns
        if feature != "operational_status"
    ],
}

results = []


print("\nSTEP 4: Training temporal models")

for variant_name, selected_features in variants.items():

    print(
        f"\nVARIANT: {variant_name} "
        f"({len(selected_features)} features)"
    )

    selected_positions = np.array(
        [
            feature_columns.index(feature)
            for feature in selected_features
        ],
        dtype=np.int64
    )

    X_train = X_all[
        np.ix_(train_indices, selected_positions)
    ]
    y_train = y_all[train_indices]

    X_validation = X_all[
        np.ix_(validation_indices, selected_positions)
    ]
    y_validation = y_all[validation_indices]

    X_test = X_all[
        np.ix_(test_indices, selected_positions)
    ]
    y_test = y_all[test_indices]

    model = LGBMClassifier(
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

    model.fit(
        X_train,
        y_train,
        eval_X=X_validation,
        eval_y=y_validation,
        eval_names=["validation"],
        eval_metric="average_precision",
        feature_name=selected_features,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                first_metric_only=True,
                verbose=False
            ),
            lgb.log_evaluation(period=50),
        ]
    )

    best_iteration = int(model.best_iteration_)

    validation_scores = model.booster_.predict(
        X_validation,
        raw_score=True,
        num_iteration=best_iteration
    )

    threshold = find_best_threshold(
        y_validation,
        validation_scores
    )

    validation_metrics = evaluate(
        y_validation,
        validation_scores,
        threshold
    )

    test_scores = model.booster_.predict(
        X_test,
        raw_score=True,
        num_iteration=best_iteration
    )

    test_metrics = evaluate(
        y_test,
        test_scores,
        threshold
    )

    print(f"Best iteration: {best_iteration}")
    print(f"Threshold: {threshold:.6f}")
    print(
        "Validation AP: "
        f"{validation_metrics['average_precision']:.6f}"
    )
    print(
        "Validation F1: "
        f"{validation_metrics['f1']:.6f}"
    )

    print("\n2019 TEST RESULTS")
    print("--------------------------------")
    print(
        f"Accuracy:  {test_metrics['accuracy']:.6f}"
    )
    print(
        f"Precision: {test_metrics['precision']:.6f}"
    )
    print(
        f"Recall:    {test_metrics['recall']:.6f}"
    )
    print(
        f"F1-score:  {test_metrics['f1']:.6f}"
    )
    print(
        f"ROC-AUC:   {test_metrics['roc_auc']:.6f}"
    )
    print(
        "PR-AUC:    "
        f"{test_metrics['average_precision']:.6f}"
    )

    print("\nCONFUSION MATRIX")
    print("--------------------------------")
    print(
        "True negatives:  "
        f"{test_metrics['true_negatives']:,}"
    )
    print(
        "False positives: "
        f"{test_metrics['false_positives']:,}"
    )
    print(
        "False negatives: "
        f"{test_metrics['false_negatives']:,}"
    )
    print(
        "True positives:  "
        f"{test_metrics['true_positives']:,}"
    )

    results.append({
        "variant": variant_name,
        "feature_count": len(selected_features),
        "train_end_year": TRAIN_END_YEAR,
        "validation_year": VALIDATION_YEAR,
        "test_year": TEST_YEAR,
        "best_iteration": best_iteration,
        "threshold": threshold,
        "validation_f1": validation_metrics["f1"],
        "validation_roc_auc": (
            validation_metrics["roc_auc"]
        ),
        "validation_average_precision": (
            validation_metrics["average_precision"]
        ),
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "roc_auc": test_metrics["roc_auc"],
        "average_precision": (
            test_metrics["average_precision"]
        ),
        "true_negatives": (
            test_metrics["true_negatives"]
        ),
        "false_positives": (
            test_metrics["false_positives"]
        ),
        "false_negatives": (
            test_metrics["false_negatives"]
        ),
        "true_positives": (
            test_metrics["true_positives"]
        ),
    })

    pd.DataFrame({
        "row_index": test_indices,
        "year": years[test_indices],
        "actual_label": y_test,
        "decision_score": test_scores,
        "predicted_label": (
            test_metrics["predictions"]
        ),
    }).to_parquet(
        OUTPUT_DIR
        / f"test_predictions_{variant_name}.parquet",
        index=False
    )

    importance = pd.DataFrame({
        "feature": selected_features,
        "gain_importance": (
            model.booster_.feature_importance(
                importance_type="gain"
            )
        ),
        "split_importance": (
            model.booster_.feature_importance(
                importance_type="split"
            )
        ),
    }).sort_values(
        by="gain_importance",
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR
        / f"feature_importance_{variant_name}.csv",
        index=False
    )

    joblib.dump(
        {
            "variant": variant_name,
            "feature_columns": selected_features,
            "model": model,
            "threshold": threshold,
            "score_type": "raw_margin",
            "train_end_year": TRAIN_END_YEAR,
            "validation_year": VALIDATION_YEAR,
            "test_year": TEST_YEAR,
        },
        OUTPUT_DIR / f"model_{variant_name}.joblib"
    )

    print("\nTOP 10 FEATURES")
    print("--------------------------------")
    print(importance.head(10).to_string(index=False))

    del (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        validation_scores,
        test_scores,
        model,
        importance,
    )
    gc.collect()


results_table = pd.DataFrame(results)

results_table.to_csv(
    OUTPUT_DIR / "temporal_results.csv",
    index=False
)

runtime = time.perf_counter() - start_time


print("\nTEMPORAL COMPARISON")
print("--------------------------------")
print(
    results_table[
        [
            "variant",
            "feature_count",
            "best_iteration",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        ]
    ].to_string(index=False)
)

print(f"\nTotal runtime: {runtime:.2f} seconds")
print(f"Results saved to: {OUTPUT_DIR.resolve()}")