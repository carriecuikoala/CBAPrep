# Datasets

This directory contains 18 benchmark dataset copies, excluding `covtype` and
`susy`. Each dataset contains `data.csv` and `info.json`, preserved byte-for-byte
from the supplied files. [manifest.json](manifest.json) records row counts,
feature counts, targets, class counts, split sizes, file sizes and SHA-256 checksums.
Feature counts exclude the target column and include any identifier columns.

| Dataset directory | CLI alias | Rows | Features | Target |
|---|---|---:|---:|---|
| `abalone` | `abalone` | 4,177 | 8 | `label` |
| `ada_prior` | `ada` | 4,562 | 14 | `label` |
| `avila` | `avila` | 20,867 | 10 | `Class` |
| `connect-4` | `connect` | 67,557 | 42 | `class` |
| `eeg` | `eeg` | 14,980 | 14 | `Class` |
| `google` | `google` | 9,367 | 8 | `Rating>4.2` |
| `house_prices` | `house` | 1,460 | 80 | `SalePrice>150k` |
| `jungle_chess_2pcs_raw_endgame_complete` | `jungle` | 44,819 | 6 | `class` |
| `microaggregation2` | `micro` | 20,000 | 20 | `class` |
| `mozilla4` | `mozilla4` | 15,545 | 5 | `state` |
| `obesity` | `obesity` | 2,111 | 16 | `NObeyesdad` |
| `page-blocks` | `page` | 5,473 | 10 | `class` |
| `pbcseq` | `pbcseq` | 1,945 | 18 | `binaryClass` |
| `pol` | `pol` | 15,000 | 48 | `binaryClass` |
| `Run_or_walk_information` | `run_or_walk` | 88,588 | 6 | `activity` |
| `shuttle` | `shuttle` | 58,000 | 9 | `class` |
| `USCensus` | `uscensus` | 32,561 | 14 | `Income` |
| `wall-robot-navigation` | `wall` | 5,456 | 24 | `Class` |

## Usage and splitting

Pass the directory name or CLI alias through `--dataset`, for example
`--dataset abalone`. Run commands from the repository root. The current search
loader converts the input with `DataConverter`, then uses `DatasetDivider` to
make approximately 40% training, 30% validation and 30% test splits with
`random_state=42`, without stratification. Integer split sizes are in the manifest.
Cost and benefit models still need to be fitted locally using the commands in
the main README.

The labels and additional metadata in `info.json` are retained as supplied.
Legacy fields such as `best_acc` are not results validated by this release.
Original source URLs, source versions, transformation histories and dataset
licenses are not yet catalogued here.

## Add your own dataset

Create `data/my_dataset/data.csv` and `data/my_dataset/info.json`.
The latter must contain the exact target column:

```json
{"label": "target"}
```

Use a CSV header and classification labels; exclude IDs or other leakage features
as appropriate. The pipeline supports classification, not regression targets.
Do not commit personal or confidential data. Run all commands from the repository
root. This directory also serves as the project-root marker used by the code.
