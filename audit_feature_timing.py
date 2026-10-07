"""Inspect timing-sensitive metadata in the V4FinBench h=1 data.

The report cannot prove leakage from the data alone. It shows whether
operational_status looks static or changes over a company's history, how it
relates to main_label, and whether positive rows use the latest company
status. Interpret status values using the dataset's source documentation.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/raw/company_years_h2.parquet"
OUTPUT_ROOT = ROOT / "results"


def main():
    started = time.perf_counter()
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Required file not found: {DATA_PATH}")

    columns = ["country", "company", "year", "operational_status", "main_label"]
    print("STEP 1: Loading timing-audit columns", flush=True)
    frame = pq.read_table(DATA_PATH, columns=columns).to_pandas()
    print(f"Rows: {len(frame):,}", flush=True)
    print("\nCOLUMN TYPES", flush=True)
    print(frame.dtypes.to_string(), flush=True)

    output_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = OUTPUT_ROOT / f"h1_feature_timing_audit_{output_name}"
    output_dir.mkdir(parents=True, exist_ok=False)

    frame["year_numeric"] = pd.to_numeric(frame["year"], errors="coerce")
    frame = frame.sort_values(
        ["country", "company", "year_numeric"],
        kind="mergesort",
    ).reset_index(drop=True)

    print("\nSTEP 2: Missing values and value counts", flush=True)
    missing = frame[columns].isna().sum().rename("missing_count").to_frame()
    missing["missing_percentage"] = missing["missing_count"] / len(frame) * 100
    print(missing.to_string(float_format=lambda value: f"{value:.4f}"), flush=True)
    missing.to_csv(output_dir / "missing_summary.csv")

    status_counts = (
        frame["operational_status"]
        .value_counts(dropna=False)
        .rename_axis("operational_status")
        .reset_index(name="rows")
    )
    status_counts["row_percentage"] = status_counts["rows"] / len(frame) * 100
    print("\nOPERATIONAL STATUS VALUES", flush=True)
    print(status_counts.head(50).to_string(index=False), flush=True)
    status_counts.to_csv(output_dir / "operational_status_counts.csv", index=False)

    print("\nSTEP 3: Status versus target", flush=True)
    status_target = pd.crosstab(
        frame["operational_status"].fillna("<MISSING>"),
        frame["main_label"],
        margins=True,
    )
    print(status_target.to_string(), flush=True)
    status_target.to_csv(output_dir / "status_target_crosstab.csv")

    status_target_rate = pd.crosstab(
        frame["operational_status"].fillna("<MISSING>"),
        frame["main_label"],
        normalize="index",
    )
    status_target_rate.to_csv(output_dir / "status_target_row_rates.csv")

    print("\nSTEP 4: Company chronology", flush=True)
    group_columns = ["country", "company"]
    grouped = frame.groupby(group_columns, sort=False, observed=True)
    company_summary = grouped.agg(
        rows=("main_label", "size"),
        first_year=("year_numeric", "min"),
        last_year=("year_numeric", "max"),
        positive_rows=("main_label", "sum"),
        distinct_statuses=("operational_status", lambda values: values.nunique(dropna=False)),
    ).reset_index()
    company_summary["status_changes"] = company_summary["distinct_statuses"] - 1
    print(company_summary[[
        "rows", "first_year", "last_year", "positive_rows",
        "distinct_statuses", "status_changes",
    ]].describe().to_string(), flush=True)
    company_summary.to_parquet(output_dir / "company_status_summary.parquet", index=False)

    latest_rows = frame.groupby(group_columns, sort=False, observed=True).tail(1).copy()
    latest_rows = latest_rows[group_columns + ["year_numeric", "operational_status"]]
    latest_rows = latest_rows.rename(
        columns={
            "year_numeric": "latest_year",
            "operational_status": "latest_operational_status",
        }
    )
    frame = frame.merge(latest_rows, on=group_columns, how="left", sort=False)
    frame["status_equals_company_latest"] = (
        frame["operational_status"].astype("string")
        == frame["latest_operational_status"].astype("string")
    )
    frame["year_is_company_latest"] = frame["year_numeric"] == frame["latest_year"]
    frame["years_before_company_latest"] = frame["latest_year"] - frame["year_numeric"]

    positive = frame[frame["main_label"] == 1]
    print("\nPOSITIVE-ROW TIMING CHECK", flush=True)
    print(f"Positive rows: {len(positive):,}", flush=True)
    print(
        f"Positive rows at company's latest year: {positive['year_is_company_latest'].mean():.4%}",
        flush=True,
    )
    print(
        f"Positive rows with status equal to company's latest status: "
        f"{positive['status_equals_company_latest'].mean():.4%}",
        flush=True,
    )
    print(
        f"All rows with status equal to company's latest status: "
        f"{frame['status_equals_company_latest'].mean():.4%}",
        flush=True,
    )
    print(
        f"Companies with at least one status change: "
        f"{(company_summary['status_changes'] > 0).mean():.4%}",
        flush=True,
    )

    positive_status = pd.crosstab(
        positive["operational_status"].fillna("<MISSING>"),
        positive["main_label"],
    )
    positive_status.to_csv(output_dir / "positive_status_counts.csv")

    timing_summary = pd.DataFrame(
        [
            {
                "rows": len(frame),
                "positive_rows": len(positive),
                "positive_at_company_latest_year": positive["year_is_company_latest"].mean(),
                "positive_status_equals_company_latest": positive[
                    "status_equals_company_latest"
                ].mean(),
                "all_status_equals_company_latest": frame[
                    "status_equals_company_latest"
                ].mean(),
                "companies": len(company_summary),
                "companies_with_status_change": int(
                    (company_summary["status_changes"] > 0).sum()
                ),
                "companies_with_status_change_percentage": (
                    company_summary["status_changes"] > 0
                ).mean(),
            }
        ]
    )
    timing_summary.to_csv(output_dir / "timing_summary.csv", index=False)

    # Save row-level fields needed for reproducible follow-up checks, without
    # copying the full 131-feature matrix.
    frame[
        group_columns
        + [
            "year_numeric",
            "operational_status",
            "main_label",
            "latest_year",
            "latest_operational_status",
            "status_equals_company_latest",
            "year_is_company_latest",
            "years_before_company_latest",
        ]
    ].to_parquet(output_dir / "row_timing_audit.parquet", index=False)

    (output_dir / "README.txt").write_text(
        "This report measures feature timing patterns; it does not prove leakage. "
        "Confirm the meaning and as-of date of operational_status with the data source.\n",
        encoding="utf-8",
    )
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "dataset": DATA_PATH.name,
                "columns": columns,
                "note": "Timing sensitivity is not equivalent to leakage. Domain documentation is required.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nFiles saved:", flush=True)
    print(output_dir / "timing_summary.csv", flush=True)
    print(output_dir / "status_target_crosstab.csv", flush=True)
    print(output_dir / "company_status_summary.parquet", flush=True)
    print(f"\nTotal runtime: {time.perf_counter() - started:.2f} seconds", flush=True)
    print(f"Results saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
