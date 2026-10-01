# User-provided datasets

No paper datasets are bundled. Create `data/my_dataset/data.csv` and
`data/my_dataset/info.json`. The latter must contain the exact target column:

```json
{"label": "target"}
```

Use a CSV header and classification labels; exclude IDs or other leakage features
as appropriate. The pipeline supports classification, not regression targets.
Do not commit personal or confidential data. Run all commands from the repository
root. This directory also serves as the project-root marker used by the code.
