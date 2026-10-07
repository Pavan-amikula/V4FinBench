from pathlib import Path
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
FOLD_PATH = Path("data/folds/h2/fold_assignments.txt")
OUTPUT_DIR = Path("results/h1_lightgbm_weight_tuning_fold0")

TARGET_COLUMN = "main_label"
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


print("STEP 1: Loading data")

feature_columns = [
    line.strip()
    for line in FEATURE_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

data = pq.read_table(
    DATA_PATH,
    columns=feature_columns + [TARGET_COLUMN]
).to_pandas()

y_all = data.pop(TARGET_COLUMN).to_numpy(dtype=np.int8)

X_all = data.to_numpy(
    dtype=np.float32,
    na_value=np.nan,
    copy=True
)

del data
gc.collect()

X_all[np.isinf(X_all)] = np.nan

folds = np.loadtxt(FOLD_PATH, dtype=np.int8)

train_indices = np.flatnonzero(
    np.isin(folds, [2, 3, 4])
)
validation_indices = np.flatnonzero(folds == 0)
test_indices = np.flatnonzero(folds == 1)

X_train = X_all[train_indices]
y_train = y_all[train_indices]

X_validation = X_all[validation_indices]
y_validation = y_all[validation_indices]

X_test = X_all[test_indices]
y_test = y_all[test_indices]

del X_all, y_all, folds
gc.collect()


negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())
full_weight = negative_count / positive_count

weight_candidates = [
    1.0,
    10.0,
    float(np.sqrt(full_weight)),
    50.0,
    100.0,
    float(full_weight),
]

print(f"Training rows: {len(y_train):,}")
print(f"Validation rows: {len(y_validation):,}")
print(f"Test rows: {len(y_test):,}")
print(f"Full imbalance ratio: {full_weight:.4f}")


print("\nSTEP 2: Testing class weights")

search_results = []

best_model = None
best_weight = None
best_iteration = None
best_validation_pr_auc = -1.0
best_threshold = None
best_validation_f1 = None

for weight in weight_candidates:

    print(f"\nTraining with class weight: {weight:.4f}")

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
        scale_pos_weight=weight,
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
        feature_name=feature_columns,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                first_metric_only=True,
                verbose=False
            )
        ]
    )

    iteration = model.best_iteration_

    validation_scores = model.booster_.predict(
        X_validation,
        raw_score=True,
        num_iteration=iteration
    )

    validation_pr_auc = average_precision_score(
        y_validation,
        validation_scores
    )

    validation_roc_auc = roc_auc_score(
        y_validation,
        validation_scores
    )

    threshold = find_best_threshold(
        y_validation,
        validation_scores
    )

    validation_predictions = (
        validation_scores >= threshold
    ).astype(np.int8)

    validation_precision = precision_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    validation_recall = recall_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    validation_f1 = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    search_results.append({
        "scale_pos_weight": weight,
        "best_iteration": iteration,
        "validation_pr_auc": validation_pr_auc,
        "validation_roc_auc": validation_roc_auc,
        "validation_precision": validation_precision,
        "validation_recall": validation_recall,
        "validation_f1": validation_f1,
        "threshold": threshold,
    })

    print(f"Best iteration: {iteration}")
    print(f"Validation PR-AUC: {validation_pr_auc:.6f}")
    print(f"Validation F1: {validation_f1:.6f}")

    if validation_pr_auc > best_validation_pr_auc:
        best_validation_pr_auc = validation_pr_auc
        best_model = model
        best_weight = weight
        best_iteration = iteration
        best_threshold = threshold
        best_validation_f1 = validation_f1


search_table = pd.DataFrame(search_results).sort_values(
    by="validation_pr_auc",
    ascending=False
)

search_table.to_csv(
    OUTPUT_DIR / "weight_search.csv",
    index=False
)

print("\nWEIGHT SEARCH RESULTS")
print("--------------------------------")
print(search_table.to_string(index=False))

print("\nSELECTED CONFIGURATION")
print("--------------------------------")
print(f"Weight: {best_weight:.4f}")
print(f"Best iteration: {best_iteration}")
print(f"Validation PR-AUC: {best_validation_pr_auc:.6f}")
print(f"Validation F1: {best_validation_f1:.6f}")
print(f"Threshold: {best_threshold:.6f}")


print("\nSTEP 3: Evaluating selected model on test data")

test_scores = best_model.booster_.predict(
    X_test,
    raw_score=True,
    num_iteration=best_iteration
)

test_predictions = (
    test_scores >= best_threshold
).astype(np.int8)

accuracy = accuracy_score(y_test, test_predictions)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(y_test, test_scores)
pr_auc = average_precision_score(y_test, test_scores)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

runtime = time.perf_counter() - start_time


print("\nTUNED LIGHTGBM TEST RESULTS")
print("--------------------------------")
print(f"Accuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-score:  {f1:.6f}")
print(f"ROC-AUC:   {roc_auc:.6f}")
print(f"PR-AUC:    {pr_auc:.6f}")

print("\nCONFUSION MATRIX")
print("--------------------------------")
print(f"True negatives:  {tn:,}")
print(f"False positives: {fp:,}")
print(f"False negatives: {fn:,}")
print(f"True positives:  {tp:,}")

print(f"\nTotal runtime: {runtime:.2f} seconds")


metrics = {
    "run_id": "h1_lightgbm_tuned_f0",
    "model": "Tuned LightGBM",
    "selected_weight": best_weight,
    "best_iteration": best_iteration,
    "threshold": best_threshold,
    "validation_pr_auc": best_validation_pr_auc,
    "validation_f1": best_validation_f1,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "runtime_seconds": runtime,
}

pd.DataFrame([metrics]).to_csv(
    OUTPUT_DIR / "metrics.csv",
    index=False
)

pd.DataFrame({
    "row_index": test_indices,
    "actual_label": y_test,
    "decision_score": test_scores,
    "predicted_label": test_predictions,
}).to_parquet(
    OUTPUT_DIR / "test_predictions.parquet",
    index=False
)

importance = pd.DataFrame({
    "feature": feature_columns,
    "gain_importance": (
        best_model.booster_.feature_importance(
            importance_type="gain"
        )
    ),
    "split_importance": (
        best_model.booster_.feature_importance(
            importance_type="split"
        )
    ),
}).sort_values(
    by="gain_importance",
    ascending=False
)

importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

joblib.dump(
    {
        "feature_columns": feature_columns,
        "model": best_model,
        "threshold": best_threshold,
        "score_type": "raw_margin",
        "selected_weight": best_weight,
    },
    OUTPUT_DIR / "model.joblib"
)

best_model.booster_.save_model(
    str(OUTPUT_DIR / "lightgbm_model.txt")
)

print("\nTOP 15 FEATURES")
print("--------------------------------")
print(importance.head(15).to_string(index=False))

print("\nSaved files:")
print(OUTPUT_DIR / "weight_search.csv")
print(OUTPUT_DIR / "metrics.csv")
print(OUTPUT_DIR / "test_predictions.parquet")
print(OUTPUT_DIR / "feature_importance.csv")
print(OUTPUT_DIR / "model.joblib")
print(OUTPUT_DIR / "lightgbm_model.txt")