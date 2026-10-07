from pathlib import Path
from datetime import datetime, timezone
import gc
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import sklearn
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

TARGET_COLUMN = "main_label"
NUMBER_OF_FOLDS = 5
RANDOM_SEED = 42

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
OUTPUT_DIR = Path("results") / f"h1_sgd_logistic_cv_{timestamp}"
MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=False)

start_time = time.perf_counter()


def find_best_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        scores,
    )

    if len(thresholds) == 0:
        return 0.0

    precision = precision[:-1]
    recall = recall[:-1]

    f1_values = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(recall),
        where=(precision + recall) != 0,
    )

    return float(thresholds[int(np.argmax(f1_values))])


def calculate_metrics(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(np.int8)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_true, scores),
        "average_precision": average_precision_score(y_true, scores),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }, predictions


print("STEP 1: Loading feature names and folds")

feature_columns = [
    line.strip()
    for line in FEATURE_PATH.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

folds = np.loadtxt(FOLD_PATH, dtype=np.int8)

if not set(np.unique(folds)).issubset(set(range(NUMBER_OF_FOLDS))):
    raise ValueError("Fold file contains unexpected fold numbers.")

print(f"Features: {len(feature_columns)}")
print(f"Fold assignments: {len(folds):,}")


print("\nSTEP 2: Loading dataset")

data = pq.read_table(
    DATA_PATH,
    columns=feature_columns + [TARGET_COLUMN],
).to_pandas()

if len(data) != len(folds):
    raise ValueError("Dataset rows do not match fold assignments.")

y_all = data.pop(TARGET_COLUMN).to_numpy(dtype=np.int8)
X_all = data.to_numpy(
    dtype=np.float32,
    na_value=np.nan,
    copy=True,
)

del data
gc.collect()

infinite_count = int(np.isinf(X_all).sum())
X_all[np.isinf(X_all)] = np.nan

print(f"Rows: {len(X_all):,}")
print(f"Features: {X_all.shape[1]}")
print(f"Infinite values changed to missing: {infinite_count:,}")


print("\nSTEP 3: Running five grouped fold rotations")

fold_metric_rows = []
prediction_tables = []

for validation_fold in range(NUMBER_OF_FOLDS):
    rotation_start = time.perf_counter()
    test_fold = (validation_fold + 1) % NUMBER_OF_FOLDS
    train_folds = [
        fold
        for fold in range(NUMBER_OF_FOLDS)
        if fold not in {validation_fold, test_fold}
    ]

    train_indices = np.flatnonzero(np.isin(folds, train_folds))
    validation_indices = np.flatnonzero(folds == validation_fold)
    test_indices = np.flatnonzero(folds == test_fold)

    X_train = X_all[train_indices]
    y_train = y_all[train_indices]

    X_validation = X_all[validation_indices]
    y_validation = y_all[validation_indices]

    X_test = X_all[test_indices]
    y_test = y_all[test_indices]

    print(
        f"\nROTATION {validation_fold + 1}/{NUMBER_OF_FOLDS}: "
        f"train={train_folds}, validation={validation_fold}, "
        f"test={test_fold}"
    )
    print(
        f"  Rows: train={len(y_train):,}, "
        f"validation={len(y_validation):,}, test={len(y_test):,}"
    )

    imputer = SimpleImputer(
        strategy="median",
        copy=False,
        keep_empty_features=True,
    )
    scaler = StandardScaler(copy=False)

    X_train = imputer.fit_transform(X_train)
    X_validation = imputer.transform(X_validation)
    X_test = imputer.transform(X_test)

    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        class_weight="balanced",
        max_iter=200,
        tol=0.001,
        average=True,
        random_state=RANDOM_SEED,
    )

    model.fit(X_train, y_train)

    validation_scores = model.decision_function(X_validation)
    threshold = find_best_threshold(y_validation, validation_scores)

    validation_metrics, _ = calculate_metrics(
        y_validation,
        validation_scores,
        threshold,
    )

    test_scores = model.decision_function(X_test)
    test_metrics, test_predictions = calculate_metrics(
        y_test,
        test_scores,
        threshold,
    )

    rotation_runtime = time.perf_counter() - rotation_start

    fold_metric_rows.append({
        "validation_fold": validation_fold,
        "test_fold": test_fold,
        "train_folds": ",".join(str(value) for value in train_folds),
        "training_rows": len(y_train),
        "validation_rows": len(y_validation),
        "test_rows": len(y_test),
        "training_positive_rows": int(y_train.sum()),
        "validation_positive_rows": int(y_validation.sum()),
        "test_positive_rows": int(y_test.sum()),
        "training_epochs": int(model.n_iter_),
        "threshold": threshold,
        "validation_f1": validation_metrics["f1"],
        "validation_average_precision": validation_metrics[
            "average_precision"
        ],
        **test_metrics,
        "runtime_seconds": rotation_runtime,
    })

    prediction_tables.append(pd.DataFrame({
        "row_index": test_indices,
        "validation_fold": validation_fold,
        "test_fold": test_fold,
        "actual_label": y_test,
        "decision_score": test_scores,
        "predicted_label": test_predictions,
        "threshold": threshold,
    }))

    joblib.dump(
        {
            "paper_horizon": 1,
            "prediction_task": "one-year-ahead financial distress",
            "feature_columns": feature_columns,
            "imputer": imputer,
            "scaler": scaler,
            "model": model,
            "threshold": threshold,
            "score_type": "decision_function",
            "validation_fold": validation_fold,
            "test_fold": test_fold,
            "train_folds": train_folds,
        },
        MODEL_DIR / f"model_validation{validation_fold}_test{test_fold}.joblib",
    )

    print(f"  Epochs: {model.n_iter_}")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Validation F1: {validation_metrics['f1']:.6f}")
    print(f"  Test precision: {test_metrics['precision']:.6f}")
    print(f"  Test recall: {test_metrics['recall']:.6f}")
    print(f"  Test F1: {test_metrics['f1']:.6f}")
    print(f"  Test ROC-AUC: {test_metrics['roc_auc']:.6f}")
    print(f"  Test PR-AUC: {test_metrics['average_precision']:.6f}")

    del (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
        validation_scores,
        test_scores,
        test_predictions,
        model,
        imputer,
        scaler,
    )
    gc.collect()


fold_metrics = pd.DataFrame(fold_metric_rows)
fold_metrics.to_csv(
    OUTPUT_DIR / "fold_metrics.csv",
    index=False,
)

all_test_predictions = pd.concat(
    prediction_tables,
    ignore_index=True,
).sort_values("row_index")

if len(all_test_predictions) != len(X_all):
    raise ValueError(
        "Out-of-fold predictions do not cover every dataset row exactly once."
    )

if all_test_predictions["row_index"].duplicated().any():
    raise ValueError("Duplicate out-of-fold row indices were found.")

all_test_predictions.to_parquet(
    OUTPUT_DIR / "out_of_fold_test_predictions.parquet",
    index=False,
)


summary_metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
]

summary_rows = []
for metric in summary_metrics:
    summary_rows.append({
        "metric": metric,
        "mean": float(fold_metrics[metric].mean()),
        "std": float(fold_metrics[metric].std(ddof=1)),
        "minimum": float(fold_metrics[metric].min()),
        "maximum": float(fold_metrics[metric].max()),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(
    OUTPUT_DIR / "summary.csv",
    index=False,
)


print("\nFIVE-FOLD SGD LOGISTIC REGRESSION RESULTS")
print("--------------------------------")
display_columns = [
    "validation_fold",
    "test_fold",
    "training_epochs",
    "threshold",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
]
print(fold_metrics[display_columns].to_string(index=False))

print("\nMEAN AND SAMPLE STANDARD DEVIATION")
print("--------------------------------")
print(summary[["metric", "mean", "std"]].to_string(index=False))


runtime_seconds = time.perf_counter() - start_time

metadata = {
    "run_id": OUTPUT_DIR.name,
    "model": "SGD Logistic Regression",
    "paper_horizon": 1,
    "prediction_task": "one-year-ahead financial distress",
    "dataset": str(DATA_PATH),
    "feature_count": len(feature_columns),
    "number_of_folds": NUMBER_OF_FOLDS,
    "fold_protocol": (
        "validation=i, test=(i+1)%5, training=remaining three folds"
    ),
    "score_type": "decision_function",
    "threshold_selection": "validation F1 maximization",
    "test_threshold_retuning": False,
    "random_seed": RANDOM_SEED,
    "runtime_seconds": runtime_seconds,
    "python_version": platform.python_version(),
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "pyarrow_version": pyarrow.__version__,
    "scikit_learn_version": sklearn.__version__,
}

(OUTPUT_DIR / "metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print(f"\nTotal runtime: {runtime_seconds:.2f} seconds")
print(f"Results saved to: {OUTPUT_DIR.resolve()}")
print("\nFiles created:")
print(OUTPUT_DIR / "fold_metrics.csv")
print(OUTPUT_DIR / "summary.csv")
print(OUTPUT_DIR / "out_of_fold_test_predictions.parquet")
print(OUTPUT_DIR / "metadata.json")
print(MODEL_DIR)
