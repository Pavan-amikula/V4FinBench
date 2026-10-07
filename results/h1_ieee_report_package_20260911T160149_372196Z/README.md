# V4FinBench IEEE Report Package

This package contains the verified tables, figures, metadata, and notes needed to write the final IEEE-format project report.

It intentionally excludes raw company data, out-of-fold prediction files, and trained-model binaries.

## Main contents

- `REPORT_EVIDENCE.txt`: verified findings and interpretation rules
- `tables/final_model_comparison.csv`: complete grouped-CV comparison
- `tables/final_model_comparison.tex`: ready-to-use IEEE table
- `tables/temporal_results.csv`: 2019 temporal validation results
- `tables/leakage_audit_summary.csv`: feature-timing evidence
- `tables/shap_global_importance.csv`: global explanation values
- `figures/`: 300-DPI report figures
- `supporting_files/`: model metadata and reproducibility manifests

Public portfolio edition: row-level company cases and their explanation figures are omitted. Aggregate tables and figures are retained.
