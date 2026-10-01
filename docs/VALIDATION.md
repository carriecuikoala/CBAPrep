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
