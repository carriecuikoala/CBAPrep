# Release checks

Checked in an isolated Linux directory containing only release files using
Python 3.11 and an existing scientific Python environment. This was not a fresh
dependency installation; no full paper experiments were rerun.

Observed versions: NumPy 2.0.1, pandas 2.3.2, SciPy 1.17.1, scikit-learn 1.7.1,
Matplotlib 3.11.2, category_encoders 2.11.1 and CPU PyTorch 2.4.1.
Optional XGBoost was not exercised.

- Synthetic MaxAcc, MinCost and MaxROI: passed assertions for bounds, budget,
  target and a MaxROI ratio better than full execution.
- Searcher, automatic launcher, runtime profiler and GBDT trainer `--help`:
  successful import and argument parsing.
- No paper data or pretrained models were present in the isolated directory.

The real-data offline profiling/training/search workflow is documented from its
CLI, but is not claimed to have been reproduced by these smoke checks.

## Dataset packaging checks (2026-10-06)

Checked the 18 bundled datasets in an existing Windows environment using Python
3.10.18, NumPy 2.0.1, pandas 2.3.2, SciPy 1.15.3 and scikit-learn 1.7.1.
All 36 CSV/JSON files match the supplied files byte-for-byte by SHA-256.
All dataset aliases resolve, CSVs load, target columns exist with no missing
labels, and the current conversion and 40/30/30 split path completes for all
18 datasets. Splits have disjoint row indices and retain every converted row.
File and dataset statistics are recorded in `data/manifest.json`.

These checks validate packaging and the input/split path. They do not validate a
fresh dependency installation, full offline model fitting, baseline comparisons,
or reproduction of the paper's tables and figures.
