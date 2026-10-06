# Core release scope

Assembled from the working implementation on 2026-10-01. Original experiment
code, running processes and historical results were not modified.

Included: import dependencies of search, the automatic launcher, ZOFW adapter,
runtime profiling and benefit sampling/fitting; required operators/classifiers;
data format instructions; a synthetic optimizer example; and, as of 2026-10-06,
18 dataset CSV/metadata pairs listed in `data/README.md` with checksums in
`data/manifest.json`. The budget-aware baseline runner and batch/plotting workflow
are also included; see `BUDGET_BASELINES.md` for the four supported methods.

Excluded: other `modules/experiment` scripts, external baseline systems such as
CAPS, `covtype` and `susy`, checkpoints, trained pickle artifacts, logs, cached
results, manuscript files, SSH/editor settings,
unreferenced runners/studies, empty advanced-cleaning placeholders and standalone
operator demonstrations. Compatibility optimizers and small human/machine cost
helpers remain because retained code imports them.

## Release-only correction

The fixed-pipeline adapter previously compared MaxROI candidates by utility when
no performance floor was enforced. This branch now compares the reported ratio.
The synthetic example verifies improvement over full execution on a case where
full execution is suboptimal for gain/cost. The enforced-performance-floor
branch is unchanged. Historical paper results have not been rerun or revalidated
by this correction. Other core module names and search defaults remain intact.

The two budget-comparison scripts were copied without algorithm changes from
the working implementation. The documented preparation commands explicitly set
`--benefit_label_source train_holdout`; the sample generator's existing default
is `test`, which should not be used to fit predictors for held-out evaluation.
