# Budget-aware baseline comparison

The two experiment scripts compare CBAPrep with Random-Budget Search (RBS),
Greedy-Knapsack (GK) and Uniform-Budget Allocation (UBA). They use the legacy
six-stage operator space. CAPS and other external systems are not included.
The source scripts are preserved without algorithm changes.

## Small end-to-end run

Run these commands from the repository root with the dependencies installed.
The bundled `abalone` data is used to check the whole workflow with a small
search. The sample count and search limits below are for a functional check;
they are not the paper's experiment settings.

```bash
python -m modules.cost_calculator.extract_dataset_params --datasets abalone --step 0.1 --run_num 1 --cost_param e_dataset_method --operator_space legacy --quiet
python -m modules.benefit_predictor.predict_data_generation.run --dataset abalone --model LR --iterations 12 --cost_param e_dataset_method --operator_space legacy --benefit_label_source train_holdout --random_state 42 --quiet
python -m modules.benefit_predictor.gbr_regressor --dataset abalone --model LR --cost_param e_dataset_method --operator_space legacy --benefit_model specific --gbr_estimators 10 --random_state 42
python -m modules.experiment.run_fair_budget_workflow --datasets abalone --model LR --budgets 1.0 --cost_param e_dataset_method --benefit_model specific --optimizer zofw --search_mode optimize_all --max_samples 10 --max_combos 3 --random_samples 3 --e_grid 0.1,1.0 --execution_repeats 1 --cbaprep_restarts 1 --random_state 42 --output_tag packaging_smoke
```

The first three commands measure local runtimes, generate predictor-training
samples and fit the predictor. Their artifacts are written under
`artifacts/pkl_store/`. The last command evaluates all four methods and writes
four result rows plus summary tables and plots. Exact times and selected plans
can vary with hardware and timing noise.

Use a fresh checkout or keep smoke-test artifacts separate from full experiments:
the small run creates models with the same filenames used by larger runs.
Rerunning the three preparation commands regenerates those artifacts.

## Larger comparisons and all 18 datasets

Prepare each dataset separately using the first three commands above, replacing
`abalone` with its [dataset alias](../data/README.md). Increase `--iterations`
and `--gbr_estimators` to the intended experiment settings; 300 samples and
100 estimators are examples, not verified paper settings. Predictor fitting requires
both the dataset and downstream classifier to match the comparison.
Keep `--operator_space legacy`, `--cost_param e_dataset_method` and
`--benefit_model specific` consistent across preparation and comparison.

After preparing all 18 datasets, the following runs 18 datasets x 4 budgets x
4 methods and generates summaries:

```bash
python -m modules.experiment.run_fair_budget_workflow --all_datasets --model LR --budgets 0.3,0.5,0.7,1.0 --cost_param e_dataset_method --benefit_model specific --optimizer zofw --search_mode optimize_all --max_samples 600 --max_combos 120 --random_samples 120 --execution_repeats 3 --cbaprep_restarts 3 --random_state 42 --output_tag all18_lr
```

These are runnable configuration examples, not a verified mapping to a paper
table. The full 18-dataset comparison has not been rerun for this release.
Offline preparation can be expensive. Budgets are preparation-execution
budgets in seconds, not caps on offline fitting, search or total wall time.

Use `--methods random,greedy,uniform` or `--budget_baselines_only` to omit
CBAPrep, or `--datasets abalone,ada` to select a subset. The lower-level runner
is also available as `python -m modules.experiment.budget_aware_baselines`.

The batch runner skips an existing per-dataset/per-budget CSV unless `--force`
is supplied. Use a new `--output_tag` when changing settings. Filenames do not
encode all search settings, seeds or artifact versions, so reusing a tag may
reuse results from a different configuration. `--force` overwrites matching
result files; `--continue_on_error` continues other cases and records failures.
Run jobs sequentially in a checkout because intermediate result filenames are
shared. Preserve the exact commands and environment with each experiment.

## Selection and evaluation

The actual pipeline execution uses the common 40/30/30 train/validation/test
split with seed 42. The documented benefit-sampling command uses an additional
holdout within the training partition. The generator defaults to `test` if
`--benefit_label_source` is omitted, so explicitly retain `train_holdout` for
held-out evaluation. Existing models do not become independent of the test set
merely by changing the comparison command; regenerate their training samples.

| Method | Candidate selection |
|---|---|
| CBAPrep | Surrogate optimization followed by validation reranking across restarts, unless disabled |
| Random-Budget Search | Validation accuracy among sampled candidates passing the predicted-cost filter |
| Greedy-Knapsack | Predicted marginal benefit per additional cost |
| Uniform-Budget Allocation | Uniform budget allocation and predicted-accuracy selection |

Final accuracy and F1 are evaluated on the held-out test partition after
selection. `--execution_repeats` repeats final execution and reports medians;
it does not repeat independent dataset splits or whole searches. Model-predicted
feasibility does not guarantee measured-time feasibility. The scripts may retain
fallback or over-budget plans; inspect `pred_feasible` and measured cost before
using aggregate summaries as evidence for budget-constrained claims. Summaries
and best-accuracy counts do not automatically exclude over-budget rows.

## Outputs

Outputs are under `artifacts/experiments/budget/baselines/`:

| Output | Contents |
|---|---|
| `budget_aware_baselines_<dataset>_budget<budget>_<model>_<cost_param>_<benefit_model>_<tag>.csv` | Per-case results, used for resuming |
| `budget_aware_baselines_<tag>_<model>_<cost_param>_<benefit_model>.csv` | Combined result rows |
| Files containing `average_by_method`, `best_by_budget` or `result_tables` | Summary CSVs and Markdown tables |
| Per-dataset metric PNGs, `average_metrics` and `best_count` PNGs | Comparison plots |

`actual_acc` and `actual_f1` are test metrics. `actual_time` measures preparation,
including conversion and splitting, and excludes downstream classifier fitting
and prediction. `budget_utilization = actual_time / budget`.
`gain_per_cost = (actual_acc - no_optional_baseline_accuracy) / actual_time`;
the reference baseline uses numeric features with training-median imputation.
`search_wall_time` is reported separately. `selection_acc` is validation accuracy
when a method performs validation selection; missing values in other methods
are expected. `pred_acc`, `pred_time`, and `pred_feasible` describe model predictions.

See [VALIDATION.md](VALIDATION.md) for the exact scope of release checks.
