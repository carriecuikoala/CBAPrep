# Core release scope

Assembled from the working implementation on 2026-10-01. Original experiment
code, running processes and historical results were not modified.

Included: import dependencies of search, the automatic launcher, ZOFW adapter,
runtime profiling and benefit sampling/fitting; required operators/classifiers;
data format instructions; a synthetic optimizer example; and, as of 2026-10-06,
18 dataset CSV/metadata pairs listed in `data/README.md` with checksums in
`data/manifest.json`.

Excluded: all `modules/experiment` scripts, `covtype` and `susy`, checkpoints, trained pickle
artifacts, logs, cached results, manuscript files, SSH/editor settings,
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
