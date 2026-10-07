from pathlib import Path
import kagglehub

folder = Path("data/raw")
folder.mkdir(parents=True, exist_ok=True)

file_path = kagglehub.dataset_download(
    "sebastiantomczak10/v4-group-corporate-bankruptcy",
    path="company_years_h2.parquet",
    output_dir=str(folder),
)

print("Downloaded successfully:")
print(file_path)