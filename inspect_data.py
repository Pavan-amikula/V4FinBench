from pathlib import Path
import pyarrow.parquet as pq

file_path = Path("data/raw/company_years_h2.parquet")
parquet_file = pq.ParquetFile(file_path)

print("DATASET INFORMATION")
print("-------------------")
print(f"Rows: {parquet_file.metadata.num_rows:,}")
print(f"Columns: {parquet_file.metadata.num_columns}")
print(f"Row groups: {parquet_file.metadata.num_row_groups}")

print("\nCOLUMN NAMES")
print("-------------------")
for column in parquet_file.schema.names:
    print(column)

batch = next(parquet_file.iter_batches(batch_size=5))
sample = batch.to_pandas()

print("\nFIRST 5 RECORDS")
print("-------------------")
print(sample.to_string())