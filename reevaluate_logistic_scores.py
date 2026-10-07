from pathlib import Path
import gc
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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
FOLD_PATH = Path("data/folds/h2/fold_assignments.txt")
MODEL_PATH = Path(
    "results/h1_sgd_logistic_fold0/model.joblib"
)
OUTPUT_DIR = Path(
    "results/h1_sgd_logistic_fold0_decision_scores"
)

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


print("STEP 1: Loading saved model")

bundle = joblib.load(MODEL_PATH)

feature_columns = bundle["feature_columns"]
imputer = bundle["imputer"]
scaler = bundle["scaler"]
model = bundle["model"]


print("\nSTEP 2: Loading validation and test data")

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

validation_indices = np.flatnonzero(folds == 0)
test_indices = np.flatnonzero(folds == 1)

X_validation = X_all[validation_indices]
y_validation = y_all[validation_indices]

X_test = X_all[test_indices]
y_test = y_all[test_indices]

del X_all, y_all, folds
gc.collect()


print("\nSTEP 3: Applying saved preprocessing")

X_validation = imputer.transform(X_validation)
X_validation = scaler.transform(X_validation)

X_test = imputer.transform(X_test)
X_test = scaler.transform(X_test)


print("\nSTEP 4: Finding threshold from decision scores")

validation_scores = model.decision_function(X_validation)

best_threshold = find_best_threshold(
    y_validation,
    validation_scores
)

validation_predictions = (
    validation_scores >= best_threshold
).astype(np.int8)

validation_f1 = f1_score(
    y_validation,
    validation_predictions,
    zero_division=0
)


print("\nSTEP 5: Evaluating test data")

test_scores = model.decision_function(X_test)

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


print("\nDECISION-SCORE RESULTS")
print("--------------------------------")
print(f"Threshold:     {best_threshold:.6f}")
print(f"Validation F1: {validation_f1:.6f}")
print(f"Accuracy:      {accuracy:.6f}")
print(f"Precision:     {precision:.6f}")
print(f"Recall:        {recall:.6f}")
print(f"F1-score:      {f1:.6f}")
print(f"ROC-AUC:       {roc_auc:.6f}")
print(f"PR-AUC:        {pr_auc:.6f}")

print("\nCONFUSION MATRIX")
print("--------------------------------")
print(f"True negatives:  {tn:,}")
print(f"False positives: {fp:,}")
print(f"False negatives: {fn:,}")
print(f"True positives:  {tp:,}")

print(f"\nRuntime: {runtime:.2f} seconds")


metrics = {
    "run_id": "h1_sgd_logistic_f0_decision",
    "model": "SGD Logistic Regression",
    "score_type": "decision_function",
    "threshold": best_threshold,
    "validation_f1": validation_f1,
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

bundle["threshold"] = best_threshold
bundle["score_type"] = "decision_function"

joblib.dump(
    bundle,
    OUTPUT_DIR / "model.joblib"
)

print("\nSaved corrected evaluation files:")
print(OUTPUT_DIR / "metrics.csv")
print(OUTPUT_DIR / "test_predictions.parquet")
print(OUTPUT_DIR / "model.joblib")