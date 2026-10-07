from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

file_path = Path("data/raw/company_years_h2.parquet")

columns = ["emis_id", "country", "year", "main_label"]

df = pq.read_table(
    file_path,
    columns=columns
).to_pandas()

print("TARGET ANALYSIS")
print("-------------------------")
print(f"Total records: {len(df):,}")

target_counts = df["main_label"].value_counts(
    dropna=False
).sort_index()

target_summary = pd.DataFrame({
    "Count": target_counts,
    "Percentage": (target_counts / len(df) * 100).round(4)
})

print("\nMain label distribution:")
print(target_summary)

print("\nDATASET GROUP INFORMATION")
print("-------------------------")
print(f"Unique companies: {df['emis_id'].nunique():,}")
print(f"Missing company IDs: {df['emis_id'].isna().sum():,}")
print(f"Minimum year: {df['year'].min()}")
print(f"Maximum year: {df['year'].max()}")

valid_rows = df.dropna(subset=["emis_id", "year"])
duplicate_pairs = valid_rows.duplicated(
    subset=["emis_id", "year"]
).sum()

print(f"Duplicate company-year pairs: {duplicate_pairs:,}")

print("\nRecords by country:")
country_counts = df["country"].value_counts(
    dropna=False
).sort_index()

print(country_counts)