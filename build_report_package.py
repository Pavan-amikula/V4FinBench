from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(".")
RESULTS_ROOT = Path("results")

CV_RUNS = {
    "SGD Logistic Regression": RESULTS_ROOT
    / "h1_sgd_logistic_cv_20260911T154854_748261Z",
    "XGBoost": RESULTS_ROOT
    / "h1_xgboost_cv_20260911T132045_245098Z",
    "CatBoost": RESULTS_ROOT
    / "h1_catboost_cv_20260911T135246_091514Z",
    "LightGBM": RESULTS_ROOT
    / "h1_lightgbm_cv_20260911T130235_435022Z",
}

TEMPORAL_DIR = RESULTS_ROOT / "h1_lightgbm_temporal_20260911T150313_430344Z"
LEAKAGE_AUDIT_DIR = (
    RESULTS_ROOT / "h1_final_leakage_audit_20260911T152816_570316Z"
)
FINAL_ANALYSIS_DIR = (
    RESULTS_ROOT / "h1_final_analysis_20260911T154234_409638Z"
)

FOLD_SUMMARY_PATH = Path("data/folds/h2/fold_summary.csv")
FEATURE_LIST_PATH = Path("data/processed/feature_columns.txt")

MODEL_ORDER = [
    "SGD Logistic Regression",
    "XGBoost",
    "CatBoost",
    "LightGBM",
]
METRIC_ORDER = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
]
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "average_precision": "PR-AUC",
}

EXPECTED_MEANS = {
    "SGD Logistic Regression": {
        "accuracy": 0.976663,
        "precision": 0.041424,
        "recall": 0.294886,
        "f1": 0.072509,
        "roc_auc": 0.901057,
        "average_precision": 0.028419,
    },
    "XGBoost": {
        "accuracy": 0.992993,
        "precision": 0.135026,
        "recall": 0.236553,
        "f1": 0.171640,
        "roc_auc": 0.942266,
        "average_precision": 0.098399,
    },
    "CatBoost": {
        "accuracy": 0.993125,
        "precision": 0.145440,
        "recall": 0.243387,
        "f1": 0.177016,
        "roc_auc": 0.939693,
        "average_precision": 0.102871,
    },
    "LightGBM": {
        "accuracy": 0.993147,
        "precision": 0.147777,
        "recall": 0.248551,
        "f1": 0.182673,
        "roc_auc": 0.941254,
        "average_precision": 0.103034,
    },
}

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
OUTPUT_DIR = RESULTS_ROOT / f"h1_ieee_report_package_{timestamp}"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
SUPPORT_DIR = OUTPUT_DIR / "supporting_files"

TABLE_DIR.mkdir(parents=True, exist_ok=False)
FIGURE_DIR.mkdir(parents=True, exist_ok=False)
SUPPORT_DIR.mkdir(parents=True, exist_ok=False)

start_time = time.perf_counter()


def normalize_metric_name(value):
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "f1_score": "f1",
        "f1score": "f1",
        "pr_auc": "average_precision",
        "average_precision_score": "average_precision",
        "rocauc": "roc_auc",
    }
    return aliases.get(normalized, normalized)


def find_cv_data(run_directory):
    if not run_directory.exists():
        raise FileNotFoundError(f"Missing result directory: {run_directory}")

    csv_paths = sorted(run_directory.rglob("*.csv"))

    # Prefer an already-aggregated mean/std table.
    for csv_path in csv_paths:
        try:
            table = pd.read_csv(csv_path)
        except Exception:
            continue

        normalized_columns = {
            str(column).strip().lower(): column
            for column in table.columns
        }
        if {"metric", "mean", "std"}.issubset(normalized_columns):
            metric_column = normalized_columns["metric"]
            mean_column = normalized_columns["mean"]
            std_column = normalized_columns["std"]

            values = {}
            for row in table.itertuples(index=False):
                row_mapping = dict(zip(table.columns, row))
                metric = normalize_metric_name(row_mapping[metric_column])
                if metric in METRIC_ORDER:
                    values[metric] = {
                        "mean": float(row_mapping[mean_column]),
                        "std": float(row_mapping[std_column]),
                    }

            if all(metric in values for metric in METRIC_ORDER):
                return values, csv_path, "summary_table"

    # Otherwise identify the per-fold test table and aggregate it.
    candidates = []
    for csv_path in csv_paths:
        try:
            table = pd.read_csv(csv_path)
        except Exception:
            continue

        renamed_columns = {
            column: normalize_metric_name(column)
            for column in table.columns
        }
        normalized_table = table.rename(columns=renamed_columns)

        required_metrics = set(METRIC_ORDER)
        if not required_metrics.issubset(normalized_table.columns):
            continue

        if len(normalized_table) < 5:
            continue

        fold_signals = {
            "validation_fold",
            "test_fold",
            "fold",
        }.intersection(normalized_table.columns)
        if not fold_signals:
            continue

        candidates.append((csv_path, normalized_table))

    if not candidates:
        available_names = ", ".join(path.name for path in csv_paths)
        raise ValueError(
            f"Could not identify five-fold metrics in {run_directory}. "
            f"CSV files found: {available_names}"
        )

    # Prefer a table with exactly five rows, then the shortest candidate.
    candidates.sort(
        key=lambda item: (
            0 if len(item[1]) == 5 else 1,
            len(item[1]),
            str(item[0]),
        )
    )
    csv_path, fold_table = candidates[0]
    fold_table = fold_table.iloc[:5].copy() if len(fold_table) > 5 else fold_table

    values = {
        metric: {
            "mean": float(fold_table[metric].mean()),
            "std": float(fold_table[metric].std(ddof=1)),
        }
        for metric in METRIC_ORDER
    }
    return values, csv_path, "computed_from_fold_table"


def copy_if_exists(source, destination):
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return True
    return False


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latex_escape(text):
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    result = str(text)
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


print("STEP 1: Reading five-fold model results")

comparison_rows = []
inventory_rows = []

for model_name in MODEL_ORDER:
    run_directory = CV_RUNS[model_name]
    values, source_path, source_type = find_cv_data(run_directory)

    row = {
        "model": model_name,
        "folds": 5,
        "source_directory": str(run_directory),
        "source_file": str(source_path),
        "source_type": source_type,
    }
    for metric in METRIC_ORDER:
        row[f"{metric}_mean"] = values[metric]["mean"]
        row[f"{metric}_std"] = values[metric]["std"]

        expected = EXPECTED_MEANS[model_name][metric]
        difference = abs(values[metric]["mean"] - expected)
        if difference > 5e-6:
            raise ValueError(
                f"{model_name} {metric} mean differs from the recorded "
                f"terminal output: found {values[metric]['mean']:.8f}, "
                f"expected approximately {expected:.8f}."
            )

    comparison_rows.append(row)
    inventory_rows.append({
        "artifact": f"{model_name} grouped cross-validation",
        "path": str(run_directory.resolve()),
        "status": "verified",
        "source_file": str(source_path.resolve()),
    })

comparison = pd.DataFrame(comparison_rows)
comparison["model"] = pd.Categorical(
    comparison["model"],
    categories=MODEL_ORDER,
    ordered=True,
)
comparison = comparison.sort_values("model").reset_index(drop=True)
comparison["model"] = comparison["model"].astype(str)

comparison.to_csv(
    TABLE_DIR / "final_model_comparison.csv",
    index=False,
)

long_rows = []
for row in comparison.itertuples(index=False):
    for metric in METRIC_ORDER:
        long_rows.append({
            "model": row.model,
            "metric": metric,
            "metric_label": METRIC_LABELS[metric],
            "mean": getattr(row, f"{metric}_mean"),
            "std": getattr(row, f"{metric}_std"),
        })

comparison_long = pd.DataFrame(long_rows)
comparison_long.to_csv(
    TABLE_DIR / "final_model_comparison_long.csv",
    index=False,
)

print(
    comparison[
        [
            "model",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "roc_auc_mean",
            "average_precision_mean",
        ]
    ].to_string(index=False)
)


print("\nSTEP 2: Determining metric winners")

winners = {}
for metric in METRIC_ORDER:
    winning_index = comparison[f"{metric}_mean"].idxmax()
    winners[metric] = comparison.loc[winning_index, "model"]
    print(
        f"{METRIC_LABELS[metric]}: {winners[metric]} "
        f"({comparison.loc[winning_index, f'{metric}_mean']:.6f})"
    )


print("\nSTEP 3: Creating IEEE-ready comparison tables")

latex_columns = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", r"$F_1$"),
    ("roc_auc", "ROC-AUC"),
    ("average_precision", "PR-AUC"),
]

latex_lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Five-fold grouped cross-validation results for one-year-ahead financial distress prediction. Values are mean $\pm$ sample standard deviation.}",
    r"\label{tab:cv-results}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    "Model & "
    + " & ".join(label for _, label in latex_columns)
    + r" \\",
    r"\midrule",
]

for row in comparison.itertuples(index=False):
    cells = [latex_escape(row.model)]
    for metric, _ in latex_columns:
        value = (
            f"${getattr(row, f'{metric}_mean'):.4f} "
            f"\\pm {getattr(row, f'{metric}_std'):.4f}$"
        )
        if winners[metric] == row.model:
            value = r"\textbf{" + value + "}"
        cells.append(value)
    latex_lines.append(" & ".join(cells) + r" \\")

latex_lines.extend([
    r"\bottomrule",
    r"\end{tabular}%",
    r"}",
    r"\end{table*}",
])

(TABLE_DIR / "final_model_comparison.tex").write_text(
    "\n".join(latex_lines) + "\n",
    encoding="utf-8",
)


print("\nSTEP 4: Creating final comparison figures")

colors = ["#7f7f7f", "#f58518", "#e45756", "#4c78a8"]

figure_metrics = ["f1", "average_precision", "roc_auc"]
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))

for axis, metric in zip(axes, figure_metrics):
    means = comparison[f"{metric}_mean"].to_numpy()
    standard_deviations = comparison[f"{metric}_std"].to_numpy()
    bars = axis.bar(
        comparison["model"],
        means,
        yerr=standard_deviations,
        capsize=4,
        color=colors,
    )
    axis.set_title(METRIC_LABELS[metric])
    axis.set_ylabel("Mean score")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.20)

    lower_limit = 0.0
    if metric == "roc_auc":
        lower_limit = max(0.0, float(means.min() - 0.03))
    upper_limit = min(
        1.0,
        float((means + standard_deviations).max() * 1.12 + 0.01),
    )
    axis.set_ylim(lower_limit, upper_limit)

    for bar, value in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

fig.suptitle("Five-Fold Grouped Cross-Validation Comparison")
fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "cv_model_comparison.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.6, 5.3))
x_positions = np.arange(len(comparison))
bar_width = 0.25

for offset, metric, color in zip(
    [-bar_width, 0.0, bar_width],
    ["precision", "recall", "f1"],
    ["#54a24b", "#eeca3b", "#4c78a8"],
):
    ax.bar(
        x_positions + offset,
        comparison[f"{metric}_mean"],
        width=bar_width,
        yerr=comparison[f"{metric}_std"],
        capsize=3,
        label=METRIC_LABELS[metric],
        color=color,
    )

ax.set_xticks(x_positions, labels=comparison["model"], rotation=15)
ax.set_ylabel("Mean score")
ax.set_title("Precision, Recall, and F1 Across Grouped Folds")
ax.grid(axis="y", alpha=0.20)
ax.legend()
fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "cv_precision_recall_f1.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.close(fig)


print("\nSTEP 5: Collecting temporal, audit, and explanation artifacts")

required_directories = [
    TEMPORAL_DIR,
    LEAKAGE_AUDIT_DIR,
    FINAL_ANALYSIS_DIR,
]
for directory in required_directories:
    if not directory.exists():
        raise FileNotFoundError(f"Missing required directory: {directory}")

temporal_results_path = TEMPORAL_DIR / "temporal_results.csv"
if not copy_if_exists(
    temporal_results_path,
    TABLE_DIR / "temporal_results.csv",
):
    raise FileNotFoundError(temporal_results_path)

copy_if_exists(
    TEMPORAL_DIR / "split_summary.csv",
    TABLE_DIR / "temporal_split_summary.csv",
)
copy_if_exists(
    LEAKAGE_AUDIT_DIR / "leakage_audit_summary.csv",
    TABLE_DIR / "leakage_audit_summary.csv",
)
copy_if_exists(
    LEAKAGE_AUDIT_DIR / "year_target_summary.csv",
    TABLE_DIR / "year_target_summary.csv",
)
copy_if_exists(
    LEAKAGE_AUDIT_DIR / "audit_notes.txt",
    SUPPORT_DIR / "audit_notes.txt",
)
copy_if_exists(
    LEAKAGE_AUDIT_DIR / "audit_metadata.json",
    SUPPORT_DIR / "audit_metadata.json",
)

for source_path in sorted((FINAL_ANALYSIS_DIR / "figures").glob("*.png")):
    shutil.copy2(source_path, FIGURE_DIR / source_path.name)

for source_path in sorted((FINAL_ANALYSIS_DIR / "tables").glob("*")):
    if source_path.is_file() and source_path.suffix.lower() in {
        ".csv",
        ".parquet",
    }:
        # Large full prediction audits are not needed inside the report ZIP.
        if source_path.name == "test_prediction_audit.parquet":
            continue
        shutil.copy2(source_path, TABLE_DIR / source_path.name)

copy_if_exists(
    FINAL_ANALYSIS_DIR / "analysis_summary.txt",
    SUPPORT_DIR / "analysis_summary.txt",
)
copy_if_exists(
    FINAL_ANALYSIS_DIR / "analysis_metadata.json",
    SUPPORT_DIR / "analysis_metadata.json",
)
copy_if_exists(
    FOLD_SUMMARY_PATH,
    TABLE_DIR / "fold_summary.csv",
)
copy_if_exists(
    FEATURE_LIST_PATH,
    SUPPORT_DIR / "feature_columns.txt",
)

inventory_rows.extend([
    {
        "artifact": "Temporal LightGBM evaluation",
        "path": str(TEMPORAL_DIR.resolve()),
        "status": "verified",
        "source_file": str(temporal_results_path.resolve()),
    },
    {
        "artifact": "Feature timing and leakage audit",
        "path": str(LEAKAGE_AUDIT_DIR.resolve()),
        "status": "verified",
        "source_file": str(
            (LEAKAGE_AUDIT_DIR / "leakage_audit_summary.csv").resolve()
        ),
    },
    {
        "artifact": "Final SHAP and error analysis",
        "path": str(FINAL_ANALYSIS_DIR.resolve()),
        "status": "verified",
        "source_file": str(
            (FINAL_ANALYSIS_DIR / "analysis_summary.txt").resolve()
        ),
    },
])

pd.DataFrame(inventory_rows).to_csv(
    TABLE_DIR / "experiment_inventory.csv",
    index=False,
)


print("\nSTEP 6: Creating reproducibility manifest")

script_rows = []
for script_path in sorted(PROJECT_ROOT.glob("*.py")):
    script_rows.append({
        "filename": script_path.name,
        "size_bytes": script_path.stat().st_size,
        "sha256": file_sha256(script_path),
    })

pd.DataFrame(script_rows).to_csv(
    SUPPORT_DIR / "project_script_manifest.csv",
    index=False,
)

freeze_result = subprocess.run(
    [sys.executable, "-m", "pip", "freeze"],
    capture_output=True,
    text=True,
    check=False,
)

if freeze_result.returncode == 0:
    (SUPPORT_DIR / "requirements_frozen.txt").write_text(
        freeze_result.stdout,
        encoding="utf-8",
    )
else:
    (SUPPORT_DIR / "requirements_frozen_error.txt").write_text(
        freeze_result.stderr,
        encoding="utf-8",
    )


print("\nSTEP 7: Writing final report notes")

lightgbm_row = comparison.loc[
    comparison["model"] == "LightGBM"
].iloc[0]
logistic_row = comparison.loc[
    comparison["model"] == "SGD Logistic Regression"
].iloc[0]
xgboost_row = comparison.loc[
    comparison["model"] == "XGBoost"
].iloc[0]

f1_relative_improvement = (
    lightgbm_row["f1_mean"] / logistic_row["f1_mean"] - 1.0
) * 100.0
ap_relative_improvement = (
    lightgbm_row["average_precision_mean"]
    / logistic_row["average_precision_mean"]
    - 1.0
) * 100.0
roc_difference_xgb_lgbm = (
    xgboost_row["roc_auc_mean"] - lightgbm_row["roc_auc_mean"]
)

temporal_results = pd.read_csv(temporal_results_path)
variant_column = "variant" if "variant" in temporal_results.columns else None

all_features_temporal = None
without_status_temporal = None
if variant_column:
    normalized_variants = temporal_results[variant_column].astype(str).str.lower()
    all_matches = temporal_results[
        normalized_variants == "all_features"
    ]
    no_status_matches = temporal_results[
        normalized_variants == "without_operational_status"
    ]
    if len(all_matches):
        all_features_temporal = all_matches.iloc[0]
    if len(no_status_matches):
        without_status_temporal = no_status_matches.iloc[0]

shap_path = TABLE_DIR / "shap_global_importance.csv"
top_shap_features = []
if shap_path.exists():
    shap_table = pd.read_csv(shap_path)
    if "feature" in shap_table.columns:
        top_shap_features = shap_table["feature"].head(10).astype(str).tolist()

report_lines = [
    "V4FinBench final report evidence summary",
    "",
    "Task definition:",
    "- Dataset file: company_years_h2.parquet.",
    "- Paper horizon: h=1, one-year-ahead financial distress.",
    "- Rows: 996,500.",
    "- Features: 131.",
    "- Positive records: 3,054.",
    "- Validation protocol: five company-grouped, country-preserving folds.",
    "",
    "Primary grouped cross-validation findings:",
    (
        f"- LightGBM achieved the best mean F1 "
        f"({lightgbm_row['f1_mean']:.6f}) and PR-AUC "
        f"({lightgbm_row['average_precision_mean']:.6f})."
    ),
    (
        f"- LightGBM improved mean F1 by {f1_relative_improvement:.1f}% "
        f"and mean PR-AUC by {ap_relative_improvement:.1f}% relative "
        "to SGD Logistic Regression."
    ),
    (
        f"- XGBoost had the highest mean ROC-AUC "
        f"({xgboost_row['roc_auc_mean']:.6f}), exceeding LightGBM by "
        f"only {roc_difference_xgb_lgbm:.6f}."
    ),
    "- Logistic Regression produced the highest mean recall but much lower "
    "precision, F1, and PR-AUC.",
    "",
    "Model policy:",
    "- Primary benchmark model: LightGBM with all 131 official features.",
    "- Timing sensitivity model: LightGBM without operational_status.",
    "- Do not describe the model as deployment-ready.",
]

if all_features_temporal is not None:
    report_lines.extend([
        "",
        "Temporal evaluation:",
        (
            f"- All-feature 2019 test F1: "
            f"{all_features_temporal['f1']:.6f}."
        ),
        (
            f"- All-feature 2019 test ROC-AUC: "
            f"{all_features_temporal['roc_auc']:.6f}."
        ),
        (
            f"- All-feature 2019 test PR-AUC: "
            f"{all_features_temporal['average_precision']:.6f}."
        ),
    ])

if without_status_temporal is not None:
    report_lines.append(
        f"- Without-status 2019 test F1: "
        f"{without_status_temporal['f1']:.6f}; PR-AUC: "
        f"{without_status_temporal['average_precision']:.6f}."
    )

report_lines.extend([
    "",
    "Feature-timing audit:",
    "- operational_status was constant for 99.8891% of company groups.",
    "- 99.9502% of rows equaled the company's latest status value.",
    "- The no-status sensitivity model must therefore remain in the report.",
    "- Insolvency_flag and Loss_flag changed across company-year records and "
    "did not directly reproduce the target.",
    "",
    "Top SHAP features:",
])
report_lines.extend(
    f"- {rank}. {feature}"
    for rank, feature in enumerate(top_shap_features, start=1)
)
report_lines.extend([
    "",
    "Important interpretation constraints:",
    "- Thresholds were selected on validation data and applied unchanged to "
    "test data.",
    "- The temporal test set was not used for threshold retuning.",
    "- 2020 contains zero positive labels and is not used as a temporal test "
    "year.",
    "- Country-level results show that one global threshold transfers unevenly "
    "across countries; this is reported as a limitation rather than corrected "
    "using test labels.",
])

(OUTPUT_DIR / "REPORT_EVIDENCE.txt").write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

manifest = {
    "project": "V4FinBench one-year-ahead financial distress prediction",
    "created_utc": timestamp,
    "completion_status": "modeling and analysis complete; report pending",
    "primary_model": "LightGBM with all 131 features",
    "sensitivity_model": "LightGBM without operational_status",
    "cv_winners": winners,
    "source_directories": {
        model: str(path)
        for model, path in CV_RUNS.items()
    },
    "temporal_directory": str(TEMPORAL_DIR),
    "leakage_audit_directory": str(LEAKAGE_AUDIT_DIR),
    "analysis_directory": str(FINAL_ANALYSIS_DIR),
    "python_version": platform.python_version(),
    "package_contains_model_binaries": False,
    "package_contains_raw_company_data": False,
}

(OUTPUT_DIR / "report_package_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

readme_lines = [
    "# V4FinBench IEEE Report Package",
    "",
    "This package contains the verified tables, figures, metadata, and notes "
    "needed to write the final IEEE-format project report.",
    "",
    "It intentionally excludes raw company data, out-of-fold prediction "
    "files, and trained-model binaries.",
    "",
    "## Main contents",
    "",
    "- `REPORT_EVIDENCE.txt`: verified findings and interpretation rules",
    "- `tables/final_model_comparison.csv`: complete grouped-CV comparison",
    "- `tables/final_model_comparison.tex`: ready-to-use IEEE table",
    "- `tables/temporal_results.csv`: 2019 temporal validation results",
    "- `tables/leakage_audit_summary.csv`: feature-timing evidence",
    "- `tables/shap_global_importance.csv`: global explanation values",
    "- `figures/`: 300-DPI report figures",
    "- `supporting_files/`: model metadata and reproducibility manifests",
]

(OUTPUT_DIR / "README.md").write_text(
    "\n".join(readme_lines) + "\n",
    encoding="utf-8",
)


print("\nSTEP 8: Creating upload-ready ZIP")

runtime_seconds = time.perf_counter() - start_time
manifest["package_build_runtime_seconds"] = runtime_seconds
(OUTPUT_DIR / "report_package_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

archive_path = Path(
    shutil.make_archive(
        str(OUTPUT_DIR),
        "zip",
        root_dir=OUTPUT_DIR,
    )
)

print(f"\nTotal runtime: {runtime_seconds:.2f} seconds")
print(f"Report package folder: {OUTPUT_DIR.resolve()}")
print(f"Upload-ready ZIP: {archive_path.resolve()}")
print(f"ZIP size: {archive_path.stat().st_size / (1024 * 1024):.2f} MB")
print("\nUpload this ZIP in the chat for the final IEEE LaTeX report.")
