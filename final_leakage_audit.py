from pathlib import Path
from datetime import datetime, timezone
import json
import platform
import time

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score


DATA_PATH = Path("data/raw/company_years_h2.parquet")
OUTPUT_ROOT = Path("results")

TARGET_COLUMN = "main_label"
GROUP_COLUMNS = ["country", "company"]
AUDIT_COLUMNS = [
    "operational_status",
    "Insolvency_flag",
    "Loss_flag",
    "year",
]

# company_years_h2.parquet is V4FinBench paper horizon h=1:
# one-year-ahead financial-distress prediction.
PAPER_HORIZON = 1
TRAINING_END_YEAR = 2017
VALIDATION_YEAR = 2018
TEST_YEAR = 2019
UNRESOLVED_YEAR = 2020

timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
OUTPUT_DIR = OUTPUT_ROOT / f"h1_final_leakage_audit_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

start_time = time.perf_counter()


def safe_percentage(numerator, denominator):
    if denominator == 0:
        return np.nan
    return 100.0 * float(numerator) / float(denominator)


def display_value(value):
    if pd.isna(value):
        return "<MISSING>"
    return str(value)


print("STEP 1: Verifying audit columns")

schema_names = pq.ParquetFile(DATA_PATH).schema.names
required_columns = GROUP_COLUMNS + [TARGET_COLUMN] + AUDIT_COLUMNS
missing_columns = [
    column for column in required_columns
    if column not in schema_names
]

if missing_columns:
    raise ValueError(
        "Required columns are missing: "
        + ", ".join(missing_columns)
    )

print("Dataset file: company_years_h2.parquet")
print("Paper horizon: h=1 (one-year-ahead distress)")
print(f"Columns checked: {', '.join(AUDIT_COLUMNS)}")


print("\nSTEP 2: Loading audit data")

data = pq.read_table(
    DATA_PATH,
    columns=required_columns,
).to_pandas()

if data[TARGET_COLUMN].isna().any():
    raise ValueError("Missing target labels were found.")

if not set(data[TARGET_COLUMN].unique()).issubset({0, 1}):
    raise ValueError("The target column is not binary.")

data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(np.int8)

print(f"Rows loaded: {len(data):,}")
print(
    "Company-country groups: "
    f"{data[GROUP_COLUMNS].drop_duplicates().shape[0]:,}"
)


print("\nSTEP 3: Checking year and horizon structure")

year_summary = (
    data.groupby("year", dropna=False)[TARGET_COLUMN]
    .agg(rows="size", positive_rows="sum")
    .reset_index()
)
year_summary["negative_rows"] = (
    year_summary["rows"] - year_summary["positive_rows"]
)
year_summary["positive_percentage"] = (
    100.0 * year_summary["positive_rows"] / year_summary["rows"]
)

print(year_summary.to_string(index=False))

if TEST_YEAR not in set(data["year"].unique()):
    raise ValueError(f"Expected test year {TEST_YEAR} is absent.")

if UNRESOLVED_YEAR in set(data["year"].unique()):
    unresolved_positive_count = int(
        data.loc[
            data["year"] == UNRESOLVED_YEAR,
            TARGET_COLUMN,
        ].sum()
    )
    print(
        f"\n{UNRESOLVED_YEAR} positive labels: "
        f"{unresolved_positive_count:,}"
    )


print("\nSTEP 4: Auditing feature timing and target association")
print(
    "Target-association checks use only development years "
    f"through {VALIDATION_YEAR}."
)
print(f"The {TEST_YEAR} labels are not used for feature decisions.")

data = data.sort_values(
    GROUP_COLUMNS + ["year"],
    kind="mergesort",
).reset_index(drop=True)

development = data[data["year"] <= VALIDATION_YEAR].copy()
development["development_split"] = np.where(
    development["year"] <= TRAINING_END_YEAR,
    "training_2006_2017",
    "validation_2018",
)

audit_rows = []
value_target_rows = []
year_profile_rows = []
finding_lines = []

grouped = data.groupby(GROUP_COLUMNS, sort=False, observed=True)
company_group_count = grouped.ngroups

for column in AUDIT_COLUMNS:
    print(f"\nCOLUMN: {column}")

    series = data[column]
    non_missing_count = int(series.notna().sum())
    missing_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))

    group_unique_counts = grouped[column].nunique(dropna=True)
    groups_with_data = int((group_unique_counts > 0).sum())
    constant_groups = int(
        ((group_unique_counts <= 1) & (group_unique_counts > 0)).sum()
    )
    changing_groups = int((group_unique_counts > 1).sum())

    latest_value = grouped[column].transform("last")
    comparable_latest = series.notna() & latest_value.notna()
    rows_equal_latest = int(
        series[comparable_latest].eq(
            latest_value[comparable_latest]
        ).sum()
    )

    previous_value = grouped[column].shift(1)
    comparable_previous = series.notna() & previous_value.notna()
    value_changes = int(
        series[comparable_previous].ne(
            previous_value[comparable_previous]
        ).sum()
    )

    numeric_development = pd.to_numeric(
        development[column],
        errors="coerce",
    )
    metric_mask = numeric_development.notna()
    metric_y = development.loc[
        metric_mask,
        TARGET_COLUMN,
    ].to_numpy(dtype=np.int8)
    metric_x = numeric_development.loc[
        metric_mask
    ].to_numpy(dtype=np.float64)

    raw_roc_auc = np.nan
    oriented_roc_auc = np.nan
    raw_average_precision = np.nan
    oriented_average_precision = np.nan
    best_orientation = "not_available"
    exact_target_match_percentage = np.nan
    inverse_target_match_percentage = np.nan

    if (
        len(metric_y) > 0
        and np.unique(metric_y).size == 2
        and np.unique(metric_x).size >= 2
    ):
        raw_roc_auc = float(roc_auc_score(metric_y, metric_x))
        inverse_roc_auc = float(roc_auc_score(metric_y, -metric_x))
        raw_average_precision = float(
            average_precision_score(metric_y, metric_x)
        )
        inverse_average_precision = float(
            average_precision_score(metric_y, -metric_x)
        )

        if inverse_average_precision > raw_average_precision:
            best_orientation = "lower_values_indicate_distress"
            oriented_average_precision = inverse_average_precision
            oriented_roc_auc = inverse_roc_auc
        else:
            best_orientation = "higher_values_indicate_distress"
            oriented_average_precision = raw_average_precision
            oriented_roc_auc = raw_roc_auc

        unique_numeric_values = set(np.unique(metric_x).tolist())
        if unique_numeric_values.issubset({0.0, 1.0}):
            exact_target_match_percentage = safe_percentage(
                np.sum(metric_x.astype(np.int8) == metric_y),
                len(metric_y),
            )
            inverse_target_match_percentage = safe_percentage(
                np.sum((1 - metric_x.astype(np.int8)) == metric_y),
                len(metric_y),
            )

    for split_name, split_data in development.groupby(
        "development_split",
        sort=False,
    ):
        value_table = (
            split_data.groupby(column, dropna=False)[TARGET_COLUMN]
            .agg(rows="size", positive_rows="sum")
            .reset_index()
        )
        value_table["positive_percentage"] = (
            100.0
            * value_table["positive_rows"]
            / value_table["rows"]
        )

        for row in value_table.itertuples(index=False):
            value_target_rows.append({
                "column": column,
                "split": split_name,
                "value": display_value(getattr(row, column)),
                "rows": int(row.rows),
                "positive_rows": int(row.positive_rows),
                "positive_percentage": float(
                    row.positive_percentage
                ),
            })

    feature_year_table = (
        data.groupby("year", dropna=False)[column]
        .agg(
            rows="size",
            missing_rows=lambda values: int(values.isna().sum()),
            unique_values=lambda values: int(
                values.nunique(dropna=True)
            ),
        )
        .reset_index()
    )

    numeric_all = pd.to_numeric(data[column], errors="coerce")
    numeric_year_means = numeric_all.groupby(data["year"]).mean()

    for row in feature_year_table.itertuples(index=False):
        year_value = row.year
        year_profile_rows.append({
            "column": column,
            "year": year_value,
            "rows": int(row.rows),
            "missing_rows": int(row.missing_rows),
            "unique_values": int(row.unique_values),
            "numeric_mean": float(
                numeric_year_means.get(year_value, np.nan)
            ),
        })

    constant_group_percentage = safe_percentage(
        constant_groups,
        groups_with_data,
    )
    changing_group_percentage = safe_percentage(
        changing_groups,
        groups_with_data,
    )
    rows_equal_latest_percentage = safe_percentage(
        rows_equal_latest,
        int(comparable_latest.sum()),
    )

    audit_rows.append({
        "column": column,
        "dtype": str(series.dtype),
        "rows": len(data),
        "missing_rows": missing_count,
        "missing_percentage": safe_percentage(
            missing_count,
            len(data),
        ),
        "unique_values": unique_count,
        "company_groups": company_group_count,
        "company_groups_with_data": groups_with_data,
        "constant_company_groups": constant_groups,
        "constant_company_groups_percentage": (
            constant_group_percentage
        ),
        "changing_company_groups": changing_groups,
        "changing_company_groups_percentage": (
            changing_group_percentage
        ),
        "row_to_row_value_changes": value_changes,
        "rows_equal_company_latest_percentage": (
            rows_equal_latest_percentage
        ),
        "development_raw_roc_auc": raw_roc_auc,
        "development_oriented_roc_auc": oriented_roc_auc,
        "development_raw_average_precision": (
            raw_average_precision
        ),
        "development_oriented_average_precision": (
            oriented_average_precision
        ),
        "best_orientation": best_orientation,
        "development_exact_target_match_percentage": (
            exact_target_match_percentage
        ),
        "development_inverse_target_match_percentage": (
            inverse_target_match_percentage
        ),
    })

    print(f"  Missing: {missing_count:,}")
    print(f"  Unique values: {unique_count:,}")
    print(
        "  Company groups constant: "
        f"{constant_group_percentage:.4f}%"
    )
    print(
        "  Rows equal company latest value: "
        f"{rows_equal_latest_percentage:.4f}%"
    )
    print(f"  Row-to-row changes: {value_changes:,}")

    if not np.isnan(oriented_average_precision):
        print(
            "  Development-only univariate AP: "
            f"{oriented_average_precision:.6f}"
        )
        print(
            "  Development-only univariate ROC-AUC: "
            f"{oriented_roc_auc:.6f}"
        )

    if column == "operational_status":
        if (
            constant_group_percentage >= 99.0
            and rows_equal_latest_percentage >= 99.0
        ):
            finding_lines.append(
                "operational_status behaves almost entirely as a "
                "company-level snapshot. Its historical point-in-time "
                "availability cannot be established from the parquet "
                "values alone. Keep the no-status sensitivity result "
                "in the final report."
            )

    if (
        not np.isnan(exact_target_match_percentage)
        and max(
            exact_target_match_percentage,
            inverse_target_match_percentage,
        ) >= 99.9
    ):
        finding_lines.append(
            f"{column} almost exactly reproduces the development target "
            "or its inverse. Treat it as a possible direct-leakage field "
            "until its row-level construction is confirmed."
        )


audit_summary = pd.DataFrame(audit_rows)
value_target_rates = pd.DataFrame(value_target_rows)
feature_year_profile = pd.DataFrame(year_profile_rows)

audit_summary.to_csv(
    OUTPUT_DIR / "leakage_audit_summary.csv",
    index=False,
)
value_target_rates.to_csv(
    OUTPUT_DIR / "development_value_target_rates.csv",
    index=False,
)
feature_year_profile.to_csv(
    OUTPUT_DIR / "feature_year_profile.csv",
    index=False,
)
year_summary.to_csv(
    OUTPUT_DIR / "year_target_summary.csv",
    index=False,
)


print("\nSTEP 5: Audit summary")

print_columns = [
    "column",
    "unique_values",
    "constant_company_groups_percentage",
    "rows_equal_company_latest_percentage",
    "row_to_row_value_changes",
    "development_oriented_roc_auc",
    "development_oriented_average_precision",
    "best_orientation",
]

print(audit_summary[print_columns].to_string(index=False))

if not finding_lines:
    finding_lines.append(
        "No field automatically triggered the script's strongest "
        "structural warning. This does not prove point-in-time safety; "
        "feature provenance must still be described in the report."
    )

notes = [
    "V4FinBench final leakage audit",
    "",
    "Confirmed task mapping:",
    "- company_years_h2.parquet is paper horizon h=1.",
    "- The task predicts financial distress one year ahead.",
    "",
    "Decision boundary:",
    f"- Training period ends in {TRAINING_END_YEAR}.",
    f"- Validation year is {VALIDATION_YEAR}.",
    f"- Test year is {TEST_YEAR} and is not used by this script for "
    "target-association decisions.",
    f"- {UNRESOLVED_YEAR} is retained only for structural auditing.",
    "",
    "Findings:",
]
notes.extend(f"- {line}" for line in finding_lines)
notes.extend([
    "",
    "Interpretation rule:",
    "- Strong predictive power alone is not proof of leakage.",
    "- Near-constant company metadata repeated across historical years "
    "requires a point-in-time availability caveat.",
    "- Do not change the decision threshold using the 2019 test labels.",
])

(OUTPUT_DIR / "audit_notes.txt").write_text(
    "\n".join(notes) + "\n",
    encoding="utf-8",
)

runtime_seconds = time.perf_counter() - start_time

metadata = {
    "dataset": str(DATA_PATH),
    "kaggle_horizon_file": 2,
    "paper_horizon": PAPER_HORIZON,
    "prediction_task": "one-year-ahead financial distress",
    "training_end_year": TRAINING_END_YEAR,
    "validation_year": VALIDATION_YEAR,
    "test_year": TEST_YEAR,
    "unresolved_year": UNRESOLVED_YEAR,
    "rows": int(len(data)),
    "company_country_groups": int(company_group_count),
    "audited_columns": AUDIT_COLUMNS,
    "runtime_seconds": runtime_seconds,
    "python_version": platform.python_version(),
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "pyarrow_version": pyarrow.__version__,
    "scikit_learn_version": sklearn.__version__,
}

(OUTPUT_DIR / "audit_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

print("\nAUDIT NOTES")
print("--------------------------------")
for line in finding_lines:
    print(f"- {line}")

print(f"\nTotal runtime: {runtime_seconds:.2f} seconds")
print(f"Results saved to: {OUTPUT_DIR.resolve()}")
print("\nFiles created:")
for filename in [
    "leakage_audit_summary.csv",
    "development_value_target_rates.csv",
    "feature_year_profile.csv",
    "year_target_summary.csv",
    "audit_notes.txt",
    "audit_metadata.json",
]:
    print(OUTPUT_DIR / filename)
