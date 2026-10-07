from pathlib import Path
from datetime import datetime, timezone
import json
import platform
import time

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


DATA_PATH = Path("data/raw/company_years_h2.parquet")
TEMPORAL_DIR = Path(
    "results/h1_lightgbm_temporal_20260911T150313_430344Z"
)
MODEL_PATH = TEMPORAL_DIR / "model_all_features.joblib"
TEMPORAL_RESULTS_PATH = TEMPORAL_DIR / "temporal_results.csv"

TARGET_COLUMN = "main_label"
ID_COLUMNS = ["country", "company", "year"]
VARIANT = "all_features"
VALIDATION_YEAR = 2018
TEST_YEAR = 2019

SHAP_SAMPLE_SIZE = 10_000
SHAP_SUMMARY_POINTS_PER_FEATURE = 2_000
TOP_GLOBAL_FEATURES = 20
TOP_CASE_FEATURES = 12
TOP_ERROR_ROWS_PER_TYPE = 50
RANDOM_SEED = 42
FIGURE_DPI = 300

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
OUTPUT_DIR = Path("results") / f"h1_final_analysis_{timestamp}"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR.mkdir(parents=True, exist_ok=False)
TABLE_DIR.mkdir(parents=True, exist_ok=False)

start_time = time.perf_counter()
rng = np.random.RandomState(RANDOM_SEED)


def normalize_name(value):
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def first_present(mapping, names):
    if not isinstance(mapping, dict):
        return None
    for name in names:
        if name in mapping:
            value = mapping[name]
            if value is not None:
                try:
                    if pd.isna(value):
                        continue
                except (TypeError, ValueError):
                    pass
                return value
    return None


def scalar_from_row(row, names):
    if row is None:
        return None
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return None


def safe_sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    probabilities = np.empty_like(values)
    positive_mask = values >= 0
    probabilities[positive_mask] = 1.0 / (
        1.0 + np.exp(-values[positive_mask])
    )
    negative_exp = np.exp(values[~positive_mask])
    probabilities[~positive_mask] = negative_exp / (1.0 + negative_exp)
    return probabilities


def calculate_metrics(y_true, scores, threshold):
    predictions = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
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
    }
    return metrics, predictions


def save_figure(fig, filename):
    fig.savefig(
        FIGURE_DIR / filename,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def short_feature_name(name, maximum=46):
    name = str(name)
    if len(name) <= maximum:
        return name
    return name[: maximum - 3] + "..."


def safe_metric(function, y_true, scores):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(function(y_true, scores))


print("STEP 1: Loading frozen temporal model")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not TEMPORAL_RESULTS_PATH.exists():
    raise FileNotFoundError(
        f"Temporal results not found: {TEMPORAL_RESULTS_PATH}"
    )

temporal_results = pd.read_csv(TEMPORAL_RESULTS_PATH)

variant_column = next(
    (
        column
        for column in ["variant", "model_variant", "run_id"]
        if column in temporal_results.columns
    ),
    None,
)

temporal_row = None
if variant_column is not None:
    normalized_variants = temporal_results[variant_column].map(normalize_name)
    matches = temporal_results.loc[
        normalized_variants == normalize_name(VARIANT)
    ]
    if len(matches) == 1:
        temporal_row = matches.iloc[0]
    elif len(matches) > 1:
        temporal_row = matches.iloc[0]
elif len(temporal_results) == 1:
    temporal_row = temporal_results.iloc[0]

loaded_object = joblib.load(MODEL_PATH)

if isinstance(loaded_object, dict):
    bundle = loaded_object
    model = first_present(
        bundle,
        ["model", "classifier", "estimator", "booster"],
    )
    feature_columns = first_present(
        bundle,
        ["feature_columns", "features", "feature_names"],
    )
    threshold = first_present(
        bundle,
        ["threshold", "best_threshold", "raw_score_threshold"],
    )
    best_iteration = first_present(
        bundle,
        ["best_iteration", "best_iteration_", "num_iteration"],
    )
else:
    bundle = {}
    model = loaded_object
    feature_columns = None
    threshold = None
    best_iteration = None

if model is None:
    raise ValueError("The joblib file does not contain a model.")

if isinstance(model, lgb.Booster):
    booster = model
elif hasattr(model, "booster_"):
    booster = model.booster_
elif hasattr(model, "_Booster"):
    booster = model._Booster
else:
    raise TypeError(
        "The saved model is not a LightGBM Booster or LGBMClassifier."
    )

if feature_columns is None:
    feature_columns = booster.feature_name()

feature_columns = [str(column) for column in feature_columns]

if threshold is None:
    threshold = scalar_from_row(
        temporal_row,
        ["threshold", "best_threshold", "raw_score_threshold"],
    )

if best_iteration is None and hasattr(model, "best_iteration_"):
    best_iteration = model.best_iteration_

if best_iteration is None:
    best_iteration = scalar_from_row(
        temporal_row,
        ["best_iteration", "iteration", "num_iteration"],
    )

if best_iteration is None or int(best_iteration) <= 0:
    best_iteration = booster.best_iteration

if best_iteration is None or int(best_iteration) <= 0:
    best_iteration = booster.current_iteration()

if threshold is None:
    raise ValueError(
        "Could not find the frozen raw-score threshold in either the "
        "joblib bundle or temporal_results.csv."
    )

threshold = float(threshold)
best_iteration = int(best_iteration)

if len(feature_columns) != booster.num_feature():
    raise ValueError(
        "Feature-name count does not match the saved LightGBM model: "
        f"{len(feature_columns)} versus {booster.num_feature()}."
    )

print(f"Variant: {VARIANT}")
print(f"Features: {len(feature_columns)}")
print(f"Best iteration: {best_iteration}")
print(f"Frozen raw-score threshold: {threshold:.6f}")


print("\nSTEP 2: Loading 2018 validation and 2019 test rows")

read_columns = []
for column in ID_COLUMNS + feature_columns + [TARGET_COLUMN]:
    if column not in read_columns:
        read_columns.append(column)

schema_names = pq.ParquetFile(DATA_PATH).schema.names
missing_columns = [
    column for column in read_columns
    if column not in schema_names
]
if missing_columns:
    raise ValueError(
        "Dataset is missing required columns: "
        + ", ".join(missing_columns)
    )

analysis_data = pq.read_table(
    DATA_PATH,
    columns=read_columns,
    filters=[("year", "in", [VALIDATION_YEAR, TEST_YEAR])],
).to_pandas()

validation_data = analysis_data.loc[
    analysis_data["year"] == VALIDATION_YEAR
].reset_index(drop=True)
test_data = analysis_data.loc[
    analysis_data["year"] == TEST_YEAR
].reset_index(drop=True)

if len(validation_data) == 0 or len(test_data) == 0:
    raise ValueError("Validation or test year produced zero rows.")

y_validation = validation_data[TARGET_COLUMN].to_numpy(dtype=np.int8)
y_test = test_data[TARGET_COLUMN].to_numpy(dtype=np.int8)

X_validation = validation_data[feature_columns].to_numpy(
    dtype=np.float32,
    na_value=np.nan,
    copy=True,
)
X_test = test_data[feature_columns].to_numpy(
    dtype=np.float32,
    na_value=np.nan,
    copy=True,
)

X_validation[np.isinf(X_validation)] = np.nan
X_test[np.isinf(X_test)] = np.nan

print(
    f"Validation: {len(y_validation):,} rows, "
    f"{int(y_validation.sum()):,} positives"
)
print(
    f"Testing: {len(y_test):,} rows, "
    f"{int(y_test.sum()):,} positives"
)


print("\nSTEP 3: Reproducing frozen-model metrics")

validation_scores = booster.predict(
    X_validation,
    raw_score=True,
    num_iteration=best_iteration,
)
test_scores = booster.predict(
    X_test,
    raw_score=True,
    num_iteration=best_iteration,
)
test_probabilities = safe_sigmoid(test_scores)

validation_metrics, validation_predictions = calculate_metrics(
    y_validation,
    validation_scores,
    threshold,
)
test_metrics, test_predictions = calculate_metrics(
    y_test,
    test_scores,
    threshold,
)

metrics_table = pd.DataFrame([
    {
        "split": "validation_2018",
        "threshold_source": "frozen_validation_threshold",
        "threshold": threshold,
        **validation_metrics,
    },
    {
        "split": "test_2019",
        "threshold_source": "frozen_validation_threshold",
        "threshold": threshold,
        **test_metrics,
    },
])
metrics_table.to_csv(
    TABLE_DIR / "confirmed_metrics.csv",
    index=False,
)

reproduction_rows = []
if temporal_row is not None:
    for metric_name, candidate_columns in {
        "accuracy": ["accuracy"],
        "precision": ["precision"],
        "recall": ["recall"],
        "f1": ["f1", "f1_score"],
        "roc_auc": ["roc_auc"],
        "average_precision": ["average_precision", "pr_auc"],
    }.items():
        saved_value = scalar_from_row(temporal_row, candidate_columns)
        if saved_value is None:
            continue
        reproduced_value = float(test_metrics[metric_name])
        absolute_difference = abs(
            reproduced_value - float(saved_value)
        )
        reproduction_rows.append({
            "metric": metric_name,
            "saved_value": float(saved_value),
            "reproduced_value": reproduced_value,
            "absolute_difference": absolute_difference,
            "matches_within_1e_6": absolute_difference <= 1e-6,
        })

if reproduction_rows:
    reproduction_check = pd.DataFrame(reproduction_rows)
    reproduction_check.to_csv(
        TABLE_DIR / "metric_reproduction_check.csv",
        index=False,
    )
    if not reproduction_check["matches_within_1e_6"].all():
        print(
            "WARNING: At least one reproduced metric differs from "
            "temporal_results.csv by more than 1e-6."
        )

print("\nCONFIRMED 2019 TEST RESULTS")
print("--------------------------------")
print(f"Accuracy:  {test_metrics['accuracy']:.6f}")
print(f"Precision: {test_metrics['precision']:.6f}")
print(f"Recall:    {test_metrics['recall']:.6f}")
print(f"F1-score:  {test_metrics['f1']:.6f}")
print(f"ROC-AUC:   {test_metrics['roc_auc']:.6f}")
print(f"PR-AUC:    {test_metrics['average_precision']:.6f}")
print(
    "Confusion:  "
    f"TN={test_metrics['true_negatives']:,}, "
    f"FP={test_metrics['false_positives']:,}, "
    f"FN={test_metrics['false_negatives']:,}, "
    f"TP={test_metrics['true_positives']:,}"
)


print("\nSTEP 4: Creating evaluation figures")

# Figure 1: confusion matrix
matrix = np.array([
    [test_metrics["true_negatives"], test_metrics["false_positives"]],
    [test_metrics["false_negatives"], test_metrics["true_positives"]],
])
fig, ax = plt.subplots(figsize=(6.2, 5.0))
image = ax.imshow(matrix, cmap="Blues")
fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks([0, 1], labels=["Predicted healthy", "Predicted distressed"])
ax.set_yticks([0, 1], labels=["Actual healthy", "Actual distressed"])
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")
ax.set_title("2019 Temporal Test Confusion Matrix")
for row_index in range(2):
    for column_index in range(2):
        value = int(matrix[row_index, column_index])
        color = "white" if value > matrix.max() * 0.5 else "black"
        ax.text(
            column_index,
            row_index,
            f"{value:,}",
            ha="center",
            va="center",
            color=color,
            fontsize=11,
            fontweight="bold",
        )
fig.tight_layout()
save_figure(fig, "01_confusion_matrix.png")

# Figure 2: precision-recall curve
test_precision_curve, test_recall_curve, _ = precision_recall_curve(
    y_test,
    test_scores,
)
base_rate = float(y_test.mean())
fig, ax = plt.subplots(figsize=(6.5, 5.0))
ax.plot(
    test_recall_curve,
    test_precision_curve,
    color="#1f77b4",
    linewidth=2,
    label=f"LightGBM (AP={test_metrics['average_precision']:.4f})",
)
ax.axhline(
    base_rate,
    color="#777777",
    linestyle="--",
    label=f"Random baseline ({base_rate:.4f})",
)
ax.scatter(
    test_metrics["recall"],
    test_metrics["precision"],
    color="#d62728",
    s=55,
    zorder=3,
    label="Frozen operating point",
)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("2019 Temporal Test Precision-Recall Curve")
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
save_figure(fig, "02_precision_recall_curve.png")

# Figure 3: ROC curve
false_positive_rate, true_positive_rate, _ = roc_curve(y_test, test_scores)
test_fpr = (
    test_metrics["false_positives"]
    / (
        test_metrics["false_positives"]
        + test_metrics["true_negatives"]
    )
)
fig, ax = plt.subplots(figsize=(6.5, 5.0))
ax.plot(
    false_positive_rate,
    true_positive_rate,
    color="#2ca02c",
    linewidth=2,
    label=f"LightGBM (AUC={test_metrics['roc_auc']:.4f})",
)
ax.plot([0, 1], [0, 1], color="#777777", linestyle="--")
ax.scatter(
    test_fpr,
    test_metrics["recall"],
    color="#d62728",
    s=55,
    zorder=3,
    label="Frozen operating point",
)
ax.set_xlabel("False-positive rate")
ax.set_ylabel("True-positive rate")
ax.set_title("2019 Temporal Test ROC Curve")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
save_figure(fig, "03_roc_curve.png")

# Figure 4: validation-only threshold diagnostic
validation_precision_curve, validation_recall_curve, validation_thresholds = (
    precision_recall_curve(y_validation, validation_scores)
)
validation_precision_curve = validation_precision_curve[:-1]
validation_recall_curve = validation_recall_curve[:-1]
validation_f1_curve = np.divide(
    2.0 * validation_precision_curve * validation_recall_curve,
    validation_precision_curve + validation_recall_curve,
    out=np.zeros_like(validation_recall_curve),
    where=(validation_precision_curve + validation_recall_curve) != 0,
)

if len(validation_thresholds) > 5_000:
    plot_indices = np.linspace(
        0,
        len(validation_thresholds) - 1,
        5_000,
        dtype=int,
    )
else:
    plot_indices = np.arange(len(validation_thresholds))

threshold_low = min(
    float(np.nanpercentile(validation_thresholds, 1)),
    threshold,
)
threshold_high = max(
    float(np.nanpercentile(validation_thresholds, 99)),
    threshold,
)

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.plot(
    validation_thresholds[plot_indices],
    validation_precision_curve[plot_indices],
    label="Precision",
    linewidth=1.6,
)
ax.plot(
    validation_thresholds[plot_indices],
    validation_recall_curve[plot_indices],
    label="Recall",
    linewidth=1.6,
)
ax.plot(
    validation_thresholds[plot_indices],
    validation_f1_curve[plot_indices],
    label="F1",
    linewidth=2.0,
)
ax.axvline(
    threshold,
    color="#d62728",
    linestyle="--",
    label=f"Selected threshold ({threshold:.3f})",
)
ax.set_xlim(threshold_low, threshold_high)
ax.set_xlabel("Raw decision threshold")
ax.set_ylabel("Metric value")
ax.set_title("2018 Validation Threshold Selection")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
save_figure(fig, "04_validation_threshold_diagnostic.png")

# Figure 5: raw-score distributions
finite_scores = test_scores[np.isfinite(test_scores)]
score_low = float(np.percentile(finite_scores, 0.5))
score_high = float(np.percentile(finite_scores, 99.5))
score_low = min(score_low, threshold)
score_high = max(score_high, threshold)
score_bins = np.linspace(score_low, score_high, 70)
fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.hist(
    test_scores[y_test == 0],
    bins=score_bins,
    density=True,
    alpha=0.60,
    label="Healthy",
    color="#4c78a8",
)
ax.hist(
    test_scores[y_test == 1],
    bins=score_bins,
    density=True,
    alpha=0.60,
    label="Distressed",
    color="#e45756",
)
ax.axvline(
    threshold,
    color="black",
    linestyle="--",
    label="Frozen threshold",
)
ax.set_xlabel("Raw model score")
ax.set_ylabel("Density")
ax.set_title("2019 Temporal Test Score Distribution")
ax.grid(alpha=0.20)
ax.legend()
fig.tight_layout()
save_figure(fig, "05_score_distribution.png")


print("\nSTEP 5: Computing native LightGBM SHAP contributions")

positive_indices = np.flatnonzero(y_test == 1)
negative_indices = np.flatnonzero(y_test == 0)

remaining_sample_size = max(
    0,
    SHAP_SAMPLE_SIZE - len(positive_indices),
)
if remaining_sample_size < len(negative_indices):
    sampled_negative_indices = rng.choice(
        negative_indices,
        size=remaining_sample_size,
        replace=False,
    )
else:
    sampled_negative_indices = negative_indices

shap_sample_indices = np.sort(
    np.concatenate([positive_indices, sampled_negative_indices])
)
X_shap = X_test[shap_sample_indices]
y_shap = y_test[shap_sample_indices]

positive_sample_count = int((y_shap == 1).sum())
negative_sample_count = int((y_shap == 0).sum())

positive_sampling_weight = (
    len(positive_indices) / positive_sample_count
    if positive_sample_count
    else 0.0
)
negative_sampling_weight = (
    len(negative_indices) / negative_sample_count
    if negative_sample_count
    else 0.0
)
shap_sample_weights = np.where(
    y_shap == 1,
    positive_sampling_weight,
    negative_sampling_weight,
)

contribution_matrix = booster.predict(
    X_shap,
    pred_contrib=True,
    num_iteration=best_iteration,
)

if hasattr(contribution_matrix, "toarray"):
    contribution_matrix = contribution_matrix.toarray()

contribution_matrix = np.asarray(
    contribution_matrix,
    dtype=np.float64,
)

expected_column_count = len(feature_columns) + 1
if contribution_matrix.shape[1] != expected_column_count:
    raise ValueError(
        "Unexpected SHAP contribution shape: "
        f"{contribution_matrix.shape}; expected second dimension "
        f"{expected_column_count}."
    )

shap_values = contribution_matrix[:, :-1]
expected_values = contribution_matrix[:, -1]

sample_raw_scores = booster.predict(
    X_shap,
    raw_score=True,
    num_iteration=best_iteration,
)
reconstructed_scores = expected_values + shap_values.sum(axis=1)
maximum_additivity_error = float(
    np.max(np.abs(sample_raw_scores - reconstructed_scores))
)

mean_absolute_shap = np.average(
    np.abs(shap_values),
    axis=0,
    weights=shap_sample_weights,
)
mean_signed_shap = np.average(
    shap_values,
    axis=0,
    weights=shap_sample_weights,
)
mean_absolute_shap_distressed = np.mean(
    np.abs(shap_values[y_shap == 1]),
    axis=0,
) if positive_sample_count else np.full(len(feature_columns), np.nan)
mean_absolute_shap_healthy = np.mean(
    np.abs(shap_values[y_shap == 0]),
    axis=0,
) if negative_sample_count else np.full(len(feature_columns), np.nan)
gain_importance = booster.feature_importance(importance_type="gain")
split_importance = booster.feature_importance(importance_type="split")

shap_importance = pd.DataFrame({
    "feature": feature_columns,
    "mean_absolute_shap": mean_absolute_shap,
    "mean_signed_shap": mean_signed_shap,
    "mean_absolute_shap_distressed": mean_absolute_shap_distressed,
    "mean_absolute_shap_healthy": mean_absolute_shap_healthy,
    "gain_importance": gain_importance,
    "split_importance": split_importance,
}).sort_values(
    "mean_absolute_shap",
    ascending=False,
).reset_index(drop=True)

shap_importance.to_csv(
    TABLE_DIR / "shap_global_importance.csv",
    index=False,
)

print(f"SHAP sample rows: {len(shap_sample_indices):,}")
print(f"Maximum SHAP additivity error: {maximum_additivity_error:.10f}")
print("\nTOP 15 GLOBAL SHAP FEATURES")
print("--------------------------------")
print(shap_importance.head(15).to_string(index=False))

# Figure 6: global SHAP bar chart
top_global = shap_importance.head(TOP_GLOBAL_FEATURES).iloc[::-1]
fig, ax = plt.subplots(figsize=(9.0, 7.0))
ax.barh(
    [short_feature_name(name) for name in top_global["feature"]],
    top_global["mean_absolute_shap"],
    color="#4c78a8",
)
ax.set_xlabel("Mean absolute SHAP contribution")
ax.set_ylabel("Feature")
ax.set_title("Global Feature Importance on the 2019 Temporal Test")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
save_figure(fig, "06_shap_global_importance.png")

# Figure 7: dependency-aware SHAP summary
top_summary_features = shap_importance.head(15)["feature"].tolist()
feature_to_index = {
    feature: index
    for index, feature in enumerate(feature_columns)
}

fig, ax = plt.subplots(figsize=(10.0, 7.5))

for display_position, feature in enumerate(
    reversed(top_summary_features)
):
    feature_index = feature_to_index[feature]
    feature_shap = shap_values[:, feature_index]
    feature_values = X_shap[:, feature_index].astype(np.float64)

    if len(feature_shap) > SHAP_SUMMARY_POINTS_PER_FEATURE:
        point_indices = rng.choice(
            len(feature_shap),
            size=SHAP_SUMMARY_POINTS_PER_FEATURE,
            replace=False,
        )
    else:
        point_indices = np.arange(len(feature_shap))

    point_shap = feature_shap[point_indices]
    point_values = feature_values[point_indices]
    finite_value_mask = np.isfinite(point_values)
    normalized_values = np.full(len(point_values), 0.5)

    if finite_value_mask.any():
        lower = float(np.percentile(point_values[finite_value_mask], 5))
        upper = float(np.percentile(point_values[finite_value_mask], 95))
        if upper > lower:
            normalized_values[finite_value_mask] = np.clip(
                (point_values[finite_value_mask] - lower)
                / (upper - lower),
                0,
                1,
            )

    jitter = rng.normal(0.0, 0.075, size=len(point_indices))
    ax.scatter(
        point_shap,
        display_position + jitter,
        c=normalized_values,
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        s=8,
        alpha=0.50,
        linewidths=0,
    )

ax.axvline(0.0, color="#555555", linewidth=1)
ax.set_yticks(
    np.arange(len(top_summary_features)),
    labels=[
        short_feature_name(name)
        for name in reversed(top_summary_features)
    ],
)
ax.set_xlabel("SHAP contribution to raw distress score")
ax.set_ylabel("Feature")
ax.set_title("SHAP Summary for a Stratified 2019 Test Sample")
ax.grid(axis="x", alpha=0.20)

color_map = plt.cm.ScalarMappable(
    norm=Normalize(vmin=0, vmax=1),
    cmap="coolwarm",
)
color_map.set_array([])
color_bar = fig.colorbar(color_map, ax=ax, pad=0.02)
color_bar.set_ticks([0, 1])
color_bar.set_ticklabels(["Low", "High"])
color_bar.set_label("Relative feature value")

fig.tight_layout()
save_figure(fig, "07_shap_summary.png")


print("\nSTEP 6: Building company-level error analysis")

prediction_table = test_data[ID_COLUMNS].copy()
prediction_table["actual_label"] = y_test
prediction_table["predicted_label"] = test_predictions
prediction_table["raw_score"] = test_scores
prediction_table["predicted_probability"] = test_probabilities
prediction_table["threshold"] = threshold

conditions = [
    (y_test == 1) & (test_predictions == 1),
    (y_test == 0) & (test_predictions == 1),
    (y_test == 1) & (test_predictions == 0),
]
prediction_table["outcome"] = np.select(
    conditions,
    ["true_positive", "false_positive", "false_negative"],
    default="true_negative",
)

prediction_table.to_parquet(
    TABLE_DIR / "test_prediction_audit.parquet",
    index=False,
)

country_rows = []
for country, group in prediction_table.groupby("country", dropna=False):
    group_y = group["actual_label"].to_numpy(dtype=np.int8)
    group_predictions = group["predicted_label"].to_numpy(dtype=np.int8)
    group_scores = group["raw_score"].to_numpy(dtype=np.float64)
    group_tn, group_fp, group_fn, group_tp = confusion_matrix(
        group_y,
        group_predictions,
        labels=[0, 1],
    ).ravel()

    country_rows.append({
        "country": country,
        "rows": len(group),
        "positive_rows": int(group_y.sum()),
        "predicted_positive_rows": int(group_predictions.sum()),
        "precision": precision_score(
            group_y,
            group_predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            group_y,
            group_predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            group_y,
            group_predictions,
            zero_division=0,
        ),
        "roc_auc": safe_metric(
            roc_auc_score,
            group_y,
            group_scores,
        ),
        "average_precision": safe_metric(
            average_precision_score,
            group_y,
            group_scores,
        ),
        "true_negatives": int(group_tn),
        "false_positives": int(group_fp),
        "false_negatives": int(group_fn),
        "true_positives": int(group_tp),
    })

country_summary = pd.DataFrame(country_rows).sort_values("country")
country_summary.to_csv(
    TABLE_DIR / "country_error_summary.csv",
    index=False,
)

error_selections = []

false_positives = prediction_table[
    prediction_table["outcome"] == "false_positive"
].nlargest(TOP_ERROR_ROWS_PER_TYPE, "raw_score")
false_negatives = prediction_table[
    prediction_table["outcome"] == "false_negative"
].nlargest(TOP_ERROR_ROWS_PER_TYPE, "raw_score")
true_positives = prediction_table[
    prediction_table["outcome"] == "true_positive"
].nlargest(TOP_ERROR_ROWS_PER_TYPE, "raw_score")

for selection in [false_positives, false_negatives, true_positives]:
    if len(selection):
        selected_rows = test_data.loc[selection.index, read_columns].copy()
        selected_rows["predicted_label"] = selection["predicted_label"]
        selected_rows["raw_score"] = selection["raw_score"]
        selected_rows["predicted_probability"] = selection[
            "predicted_probability"
        ]
        selected_rows["outcome"] = selection["outcome"]
        error_selections.append(selected_rows)

if error_selections:
    top_error_cases = pd.concat(error_selections, ignore_index=True)
    top_error_cases.to_csv(
        TABLE_DIR / "top_error_cases.csv",
        index=False,
    )

print("\nCOUNTRY ERROR SUMMARY")
print("--------------------------------")
print(country_summary.to_string(index=False))


print("\nSTEP 7: Explaining representative predictions")

representative_indices = {}

tp_indices = np.flatnonzero(
    (y_test == 1) & (test_predictions == 1)
)
fp_indices = np.flatnonzero(
    (y_test == 0) & (test_predictions == 1)
)
fn_indices = np.flatnonzero(
    (y_test == 1) & (test_predictions == 0)
)

if len(tp_indices):
    representative_indices["highest_confidence_true_positive"] = int(
        tp_indices[np.argmax(test_scores[tp_indices])]
    )
if len(fp_indices):
    representative_indices["highest_confidence_false_positive"] = int(
        fp_indices[np.argmax(test_scores[fp_indices])]
    )
if len(fn_indices):
    representative_indices["nearest_threshold_false_negative"] = int(
        fn_indices[np.argmax(test_scores[fn_indices])]
    )

representative_case_rows = []
representative_contribution_rows = []

for case_name, row_index in representative_indices.items():
    case_X = X_test[row_index : row_index + 1]
    case_contributions = booster.predict(
        case_X,
        pred_contrib=True,
        num_iteration=best_iteration,
    )
    if hasattr(case_contributions, "toarray"):
        case_contributions = case_contributions.toarray()
    case_contributions = np.asarray(case_contributions)[0]
    case_shap = case_contributions[:-1]
    case_expected_value = float(case_contributions[-1])

    case_meta = prediction_table.iloc[row_index]
    representative_case_rows.append({
        "case": case_name,
        "country": case_meta["country"],
        "company": case_meta["company"],
        "year": int(case_meta["year"]),
        "actual_label": int(case_meta["actual_label"]),
        "predicted_label": int(case_meta["predicted_label"]),
        "raw_score": float(case_meta["raw_score"]),
        "predicted_probability": float(
            case_meta["predicted_probability"]
        ),
        "threshold": threshold,
        "expected_value": case_expected_value,
    })

    for feature_index, feature in enumerate(feature_columns):
        representative_contribution_rows.append({
            "case": case_name,
            "feature": feature,
            "feature_value": float(case_X[0, feature_index])
            if np.isfinite(case_X[0, feature_index])
            else np.nan,
            "shap_contribution": float(case_shap[feature_index]),
            "absolute_shap_contribution": float(
                abs(case_shap[feature_index])
            ),
        })

    order = np.argsort(np.abs(case_shap))[::-1]
    top_indices = order[:TOP_CASE_FEATURES]
    other_contribution = float(case_shap[order[TOP_CASE_FEATURES:]].sum())

    plot_labels = [
        short_feature_name(feature_columns[index], maximum=40)
        for index in top_indices
    ]
    plot_values = [float(case_shap[index]) for index in top_indices]

    if len(order) > TOP_CASE_FEATURES:
        plot_labels.append("All other features")
        plot_values.append(other_contribution)

    plot_labels = plot_labels[::-1]
    plot_values = plot_values[::-1]
    plot_colors = [
        "#e45756" if value > 0 else "#4c78a8"
        for value in plot_values
    ]

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    ax.barh(plot_labels, plot_values, color=plot_colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution to raw distress score")
    ax.set_ylabel("Feature")
    ax.set_title(
        case_name.replace("_", " ").title()
        + "\n"
        + f"actual={int(y_test[row_index])}, "
        + f"predicted={int(test_predictions[row_index])}, "
        + f"score={test_scores[row_index]:.3f}"
    )
    ax.grid(axis="x", alpha=0.20)
    fig.tight_layout()
    save_figure(fig, f"case_{case_name}.png")

pd.DataFrame(representative_case_rows).to_csv(
    TABLE_DIR / "representative_cases.csv",
    index=False,
)
pd.DataFrame(representative_contribution_rows).to_csv(
    TABLE_DIR / "representative_case_contributions.csv",
    index=False,
)


print("\nSTEP 8: Comparing temporal feature variants")

comparison_metric_columns = [
    column
    for column in ["average_precision", "f1", "roc_auc"]
    if column in temporal_results.columns
]

if variant_column is not None and comparison_metric_columns:
    comparison_data = temporal_results.copy()
    comparison_data["display_variant"] = comparison_data[variant_column].map(
        lambda value: {
            "all_features": "All features",
            "without_operational_status": "Without status",
        }.get(normalize_name(value), str(value))
    )

    figure_count = len(comparison_metric_columns)
    fig, axes = plt.subplots(
        1,
        figure_count,
        figsize=(4.2 * figure_count, 4.6),
        squeeze=False,
    )

    metric_labels = {
        "average_precision": "PR-AUC",
        "f1": "F1",
        "roc_auc": "ROC-AUC",
    }

    for axis, metric in zip(axes[0], comparison_metric_columns):
        bars = axis.bar(
            comparison_data["display_variant"],
            comparison_data[metric],
            color=["#4c78a8", "#f58518"][: len(comparison_data)],
        )
        axis.set_title(metric_labels[metric])
        axis.set_ylabel("Score")
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.20)
        maximum_value = float(comparison_data[metric].max())
        axis.set_ylim(0, min(1.0, maximum_value * 1.30 + 0.01))
        for bar, value in zip(bars, comparison_data[metric]):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("2019 Temporal Feature-Timing Sensitivity")
    fig.tight_layout()
    save_figure(fig, "08_temporal_variant_comparison.png")
else:
    print(
        "Temporal comparison figure skipped because the expected metric "
        "columns were not present."
    )


print("\nSTEP 9: Saving final analysis metadata")

runtime_seconds = time.perf_counter() - start_time

metadata = {
    "analysis_type": "post-hoc frozen-model evaluation and explanation",
    "variant": VARIANT,
    "dataset": str(DATA_PATH),
    "model_path": str(MODEL_PATH),
    "paper_horizon": 1,
    "prediction_task": "one-year-ahead financial distress",
    "validation_year": VALIDATION_YEAR,
    "test_year": TEST_YEAR,
    "feature_count": len(feature_columns),
    "best_iteration": best_iteration,
    "frozen_raw_score_threshold": threshold,
    "shap_sample_size": int(len(shap_sample_indices)),
    "shap_sample_positive_rows": int(y_shap.sum()),
    "shap_sample_negative_rows": negative_sample_count,
    "global_shap_importance_sampling_weighted": True,
    "shap_additivity_maximum_absolute_error": maximum_additivity_error,
    "threshold_retuned_on_test": False,
    "random_seed": RANDOM_SEED,
    "figure_dpi": FIGURE_DPI,
    "runtime_seconds": runtime_seconds,
    "python_version": platform.python_version(),
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "pyarrow_version": pyarrow.__version__,
    "scikit_learn_version": sklearn.__version__,
    "lightgbm_version": lgb.__version__,
    "matplotlib_version": matplotlib.__version__,
}

(OUTPUT_DIR / "analysis_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

summary_lines = [
    "V4FinBench final LightGBM analysis",
    "",
    "Model policy:",
    "- Main benchmark model: all 131 official features.",
    "- Timing sensitivity: report the model without operational_status.",
    "- Do not describe either model as deployment-ready.",
    "",
    "Frozen evaluation:",
    f"- Best iteration: {best_iteration}",
    f"- Raw-score threshold: {threshold:.6f}",
    f"- 2019 PR-AUC: {test_metrics['average_precision']:.6f}",
    f"- 2019 ROC-AUC: {test_metrics['roc_auc']:.6f}",
    f"- 2019 F1: {test_metrics['f1']:.6f}",
    f"- 2019 precision: {test_metrics['precision']:.6f}",
    f"- 2019 recall: {test_metrics['recall']:.6f}",
    "",
    "Interpretation:",
    "- SHAP values explain contributions to the raw LightGBM score.",
    "- Positive SHAP values push predictions toward distress.",
    "- Negative SHAP values push predictions toward the healthy class.",
    "- All plots and error tables are post-hoc analyses; the test threshold "
    "was not changed.",
]

(OUTPUT_DIR / "analysis_summary.txt").write_text(
    "\n".join(summary_lines) + "\n",
    encoding="utf-8",
)

print(f"\nTotal runtime: {runtime_seconds:.2f} seconds")
print(f"Results saved to: {OUTPUT_DIR.resolve()}")
print("\nMain report-ready outputs:")
for filename in [
    FIGURE_DIR / "01_confusion_matrix.png",
    FIGURE_DIR / "02_precision_recall_curve.png",
    FIGURE_DIR / "03_roc_curve.png",
    FIGURE_DIR / "04_validation_threshold_diagnostic.png",
    FIGURE_DIR / "06_shap_global_importance.png",
    FIGURE_DIR / "07_shap_summary.png",
    TABLE_DIR / "confirmed_metrics.csv",
    TABLE_DIR / "shap_global_importance.csv",
    TABLE_DIR / "country_error_summary.csv",
    OUTPUT_DIR / "analysis_summary.txt",
]:
    print(filename)
