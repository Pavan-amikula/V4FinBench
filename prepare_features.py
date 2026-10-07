from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DATA_PATH = Path("data/raw/company_years_h2.parquet")
FOLD_PATH = Path("data/folds/h2/fold_assignments.txt")
OUTPUT_DIR = Path("data/processed")

TARGET_COLUMN = "main_label"

COLUMNS_TO_REMOVE = [
    "company",
    "industry",
    "link",
    "num",
    "emis_id",
    "sector_2",
    "sector_3",
    "sector_4",
    "Revenue/employee",
    "Fixed_assets/employee",
    "EBITDA/cash_flow",
]


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

parquet_file = pq.ParquetFile(DATA_PATH)
all_columns = parquet_file.schema.names

feature_columns = [
    column
    for column in all_columns
    if column not in COLUMNS_TO_REMOVE
    and column != TARGET_COLUMN
]

print("FEATURE PREPARATION")
print("-----------------------------")
print(f"Original columns: {len(all_columns)}")
print(f"Removed identifier/problem columns: {len(COLUMNS_TO_REMOVE)}")
print(f"Target column: {TARGET_COLUMN}")
print(f"Model feature columns: {len(feature_columns)}")


# Verify fold file
folds = np.loadtxt(FOLD_PATH, dtype=np.int8)

if len(folds) != parquet_file.metadata.num_rows:
    raise ValueError(
        "Fold count does not match dataset row count."
    )

print(f"Fold assignments verified: {len(folds):,}")


# Inspect 50,000 rows without loading the complete dataset
batch = next(
    parquet_file.iter_batches(
        batch_size=50_000,
        columns=feature_columns
    )
)

sample = batch.to_pandas()

numeric_columns = sample.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

non_numeric_columns = [
    column
    for column in feature_columns
    if column not in numeric_columns
]

numeric_data = sample[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

numeric_array = numeric_data.to_numpy(
    dtype=np.float64,
    na_value=np.nan
)

infinite_counts = pd.Series(
    np.isinf(numeric_array).sum(axis=0),
    index=numeric_columns
)

profile = pd.DataFrame({
    "column": feature_columns,
    "dtype": [
        str(sample[column].dtype)
        for column in feature_columns
    ],
    "missing_count_sample": [
        int(sample[column].isna().sum())
        for column in feature_columns
    ],
    "missing_percentage_sample": [
        round(sample[column].isna().mean() * 100, 2)
        for column in feature_columns
    ],
    "infinite_count_sample": [
        int(infinite_counts.get(column, 0))
        for column in feature_columns
    ],
})

profile = profile.sort_values(
    by="missing_percentage_sample",
    ascending=False
)

profile.to_csv(
    OUTPUT_DIR / "feature_profile_sample.csv",
    index=False
)

Path(OUTPUT_DIR / "feature_columns.txt").write_text(
    "\n".join(feature_columns),
    encoding="utf-8"
)


print(f"Numeric or Boolean features: {len(numeric_columns)}")
print(f"Non-numeric features: {len(non_numeric_columns)}")

if non_numeric_columns:
    print("\nNon-numeric column names:")
    for column in non_numeric_columns:
        print("-", column)

print("\nTOP 15 COLUMNS WITH MISSING VALUES")
print("------------------------------------")
print(profile.head(15).to_string(index=False))

print("\nFiles created:")
print(OUTPUT_DIR / "feature_columns.txt")
print(OUTPUT_DIR / "feature_profile_sample.csv")

print("\nFeature preparation inspection completed.")