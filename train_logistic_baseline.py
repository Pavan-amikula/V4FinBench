from pathlib import Path
import gc
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
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
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/raw/company_years_h2.parquet")
FEATURE_PATH = Path("data/processed/feature_columns.txt")
FOLD_PATH = Path("data/folds/h2/fold_assignments.txt")
OUTPUT_DIR = Path("results/h1_sgd_logistic_fold0")

TARGET_COLUMN = "main_label"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

start_time = time.perf_counter()


def find_best_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities
    )

    if len(thresholds) == 0:
        return 0.5

    precision = precision[:-1]
    recall = recall[:-1]

    f1_values = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(recall),
        where=(precision + recall) != 0
    )

    best_position = int(np.argmax(f1_values))
    return float(thresholds[best_position])


print("STEP 1: Reading feature names")

feature_columns = [
    line.strip()
    for line in FEATURE_PATH.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
]

print(f"Features: {len(feature_columns)}")


print("\nSTEP 2: Loading dataset")
print("This can take a few minutes.")

data = pq.read_table(
    DATA_PATH,
    columns=feature_columns + [TARGET_COLUMN]
).to_pandas()

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

print(f"Loaded rows: {len(X_all):,}")
print(f"Loaded features: {X_all.shape[1]}")
print(f"Infinite values changed to missing: {infinite_count:,}")


print("\nSTEP 3: Loading fold assignments")

folds = np.loadtxt(
    FOLD_PATH,
    dtype=np.int8
)

if len(folds) != len(X_all):
    raise ValueError(
        "Fold assignments do not match dataset rows."
    )


# Experiment rotation 0
train_indices = np.flatnonzero(
    np.isin(folds, [2, 3, 4])
)

validation_indices = np.flatnonzero(
    folds == 0
)

test_indices = np.flatnonzero(
    folds == 1
)


X_train = X_all[train_indices]
y_train = y_all[train_indices]

X_validation = X_all[validation_indices]
y_validation = y_all[validation_indices]

X_test = X_all[test_indices]
y_test = y_all[test_indices]

del X_all, y_all, folds
gc.collect()


print("\nDATA SPLIT")
print("--------------------------------")
print(
    f"Training:   {len(y_train):,} rows, "
    f"{int(y_train.sum()):,} distressed"
)
print(
    f"Validation: {len(y_validation):,} rows, "
    f"{int(y_validation.sum()):,} distressed"
)
print(
    f"Testing:    {len(y_test):,} rows, "
    f"{int(y_test.sum()):,} distressed"
)


print("\nSTEP 4: Learning training-data medians")

imputer = SimpleImputer(
    strategy="median",
    keep_empty_features=True,
    copy=False
)

X_train = imputer.fit_transform(X_train)
X_validation = imputer.transform(X_validation)
X_test = imputer.transform(X_test)


print("\nSTEP 5: Scaling features")

scaler = StandardScaler(copy=False)

X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)


print("\nSTEP 6: Training Logistic Regression")

model = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=0.0001,
    class_weight="balanced",
    max_iter=200,
    tol=0.001,
    average=True,
    random_state=42
)

model.fit(X_train, y_train)

print(f"Training epochs completed: {model.n_iter_}")


print("\nSTEP 7: Selecting threshold on validation data")

validation_probabilities = model.predict_proba(
    X_validation
)[:, 1]

best_threshold = find_best_threshold(
    y_validation,
    validation_probabilities
)

validation_predictions = (
    validation_probabilities >= best_threshold
).astype(np.int8)

validation_f1 = f1_score(
    y_validation,
    validation_predictions,
    zero_division=0
)

print(f"Selected threshold: {best_threshold:.6f}")
print(f"Validation F1: {validation_f1:.6f}")


print("\nSTEP 8: Final evaluation on untouched test data")

test_probabilities = model.predict_proba(
    X_test
)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(np.int8)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

test_pr_auc = average_precision_score(
    y_test,
    test_probabilities
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
).ravel()

runtime_seconds = time.perf_counter() - start_time


metrics = {
    "run_id": "h1_sgd_logistic_f0",
    "model": "SGD Logistic Regression",
    "horizon": 1,
    "validation_fold": 0,
    "test_fold": 1,
    "threshold": best_threshold,
    "validation_f1": validation_f1,
    "accuracy": test_accuracy,
    "precision": test_precision,
    "recall": test_recall,
    "f1": test_f1,
    "roc_auc": test_roc_auc,
    "pr_auc": test_pr_auc,
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
    "runtime_seconds": runtime_seconds,
}


print("\nFINAL TEST RESULTS")
print("--------------------------------")
print(f"Accuracy:  {test_accuracy:.6f}")
print(f"Precision: {test_precision:.6f}")
print(f"Recall:    {test_recall:.6f}")
print(f"F1-score:  {test_f1:.6f}")
print(f"ROC-AUC:   {test_roc_auc:.6f}")
print(f"PR-AUC:    {test_pr_auc:.6f}")

print("\nCONFUSION MATRIX")
print("--------------------------------")
print(f"True negatives:  {tn:,}")
print(f"False positives: {fp:,}")
print(f"False negatives: {fn:,}")
print(f"True positives:  {tp:,}")

print(f"\nTotal runtime: {runtime_seconds:.2f} seconds")


pd.DataFrame([metrics]).to_csv(
    OUTPUT_DIR / "metrics.csv",
    index=False
)

pd.DataFrame({
    "row_index": test_indices,
    "actual_label": y_test,
    "predicted_probability": test_probabilities,
    "predicted_label": test_predictions,
}).to_parquet(
    OUTPUT_DIR / "test_predictions.parquet",
    index=False
)

joblib.dump(
    {
        "feature_columns": feature_columns,
        "imputer": imputer,
        "scaler": scaler,
        "model": model,
        "threshold": best_threshold,
    },
    OUTPUT_DIR / "model.joblib"
)

print("\nSaved files:")
print(OUTPUT_DIR / "metrics.csv")
print(OUTPUT_DIR / "test_predictions.parquet")
print(OUTPUT_DIR / "model.joblib")