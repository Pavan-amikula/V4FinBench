from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DATA_PATH = Path("data/raw/company_years_h2.parquet")
OUTPUT_DIR = Path("data/folds/h2")

NUMBER_OF_FOLDS = 5
RANDOM_SEED = 42


# Create the output folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load only the columns required for fold creation
df = pq.read_table(
    DATA_PATH,
    columns=["country", "company", "year", "main_label"]
).to_pandas()


print("CREATING OFFICIAL GROUPED FOLDS")
print("--------------------------------")
print(f"Total rows: {len(df):,}")


# Check required columns
if df["country"].isna().any():
    raise ValueError("Missing values found in the country column.")

if df["company"].isna().any():
    raise ValueError("Missing values found in the company column.")


# Create deterministic folds
random_generator = np.random.RandomState(RANDOM_SEED)
folds = np.empty(len(df), dtype=np.int8)

for country, country_data in df.groupby("country", sort=True):

    companies = (
        country_data["company"]
        .dropna()
        .unique()
        .copy()
    )

    random_generator.shuffle(companies)

    company_to_fold = {
        company: index % NUMBER_OF_FOLDS
        for index, company in enumerate(companies)
    }

    assigned_folds = country_data["company"].map(company_to_fold)

    folds[country_data.index.to_numpy()] = (
        assigned_folds.to_numpy(dtype=np.int8)
    )


df["fold"] = folds


# Verify that one company-country group never appears in multiple folds
fold_check = (
    df.groupby(["country", "company"])["fold"]
    .nunique()
    .max()
)

if fold_check != 1:
    raise ValueError("Company leakage detected between folds.")


# Save the fold number corresponding to every dataset row
np.savetxt(
    OUTPUT_DIR / "fold_assignments.txt",
    folds,
    fmt="%d"
)


# Create summary
summary_rows = []

for fold_number in range(NUMBER_OF_FOLDS):

    fold_data = df[df["fold"] == fold_number]

    summary_rows.append({
        "fold": fold_number,
        "rows": len(fold_data),
        "company_groups": (
            fold_data[["country", "company"]]
            .drop_duplicates()
            .shape[0]
        ),
        "healthy_records": int(
            (fold_data["main_label"] == 0).sum()
        ),
        "distressed_records": int(
            (fold_data["main_label"] == 1).sum()
        ),
        "distressed_percentage": round(
            fold_data["main_label"].mean() * 100,
            4
        )
    })


summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT_DIR / "fold_summary.csv",
    index=False
)


print("\nFOLD SUMMARY")
print("--------------------------------")
print(summary.to_string(index=False))

print("\nCOUNTRY RECORDS IN EACH FOLD")
print("--------------------------------")
print(pd.crosstab(df["fold"], df["country"]))

print("\nLeakage check passed.")
print("Every company-country group belongs to only one fold.")

print("\nFiles created:")
print(OUTPUT_DIR / "fold_assignments.txt")
print(OUTPUT_DIR / "fold_summary.csv")

print("\nFIRST EXPERIMENT SPLIT")
print("Training: folds 2, 3 and 4")
print("Validation: fold 0")
print("Testing: fold 1")