import argparse
import contextlib
import io
import os
import sys
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split

from modules.common.dataset_utils import (
    DATASET_SHORT_NAMES,
    find_project_root,
    resolve_dataset_paths,
    load_dataset_label,
    resolve_dataset_name,
)
from modules.common.operator_space import (
    SOTA_METHOD_ADDITIONS,
    get_method_options,
    normalize_operator_space,
)


PROCESS_NAME_MAP = {
    "duplicate_detection": "duplicate detection",
    "feature_encoding": "feature coding",
    "imputation": "imputation",
    "data_normalization": "data normalization",
    "outlier_detection": "outlier detection",
    "feature_extraction": "feature extraction",
    "feature_selection": "feature selection",
}


def get_process_methods(operator_space="legacy", only_new=False):
    operator_space = normalize_operator_space(operator_space)
    if only_new:
        source = SOTA_METHOD_ADDITIONS if operator_space == "sota" else {}
    else:
        source = get_method_options(operator_space)
    methods = {
        PROCESS_NAME_MAP[stage]: values[:]
        for stage, values in source.items()
        if stage in PROCESS_NAME_MAP
    }
    if not only_new:
        methods["data conversion"] = ["data_conversion"]
        methods["dataset division"] = ["dataset_division"]
    return methods


PROCESS_METHODS = get_process_methods("legacy")


NUMERIC_REQUIRED_PROCESSES = {
    "data normalization",
    "feature selection",
    "feature extraction",
    "imputation",
    "outlier detection",
}


def _progress(cur, total, label):
    bar_len = 30
    filled = int(bar_len * cur / total) if total > 0 else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    pct = (cur / total * 100.0) if total > 0 else 0.0
    msg = f"[{bar}] {cur}/{total} ({pct:.1f}%) {label}"
    pad = " " * max(0, _progress.last_len - len(msg))
    sys.stdout.write("\r" + msg + pad)
    sys.stdout.flush()
    _progress.last_len = len(msg)


_progress.last_len = 0


def _run_method(
    pe,
    method,
    step,
    run_num,
    dataset_name,
    process_name,
    non_num_features,
    quiet,
    cost_param,
    artifact_version=None,
    save_repeat_details=False,
):
    kwargs = {
        "run_num": run_num,
        "dataset_name": dataset_name,
        "cost_param": cost_param,
        "artifact_version": artifact_version,
        "save_repeat_details": save_repeat_details,
    }
    if process_name == "feature coding":
        kwargs["non_num_features"] = non_num_features
    if process_name == "dataset division":
        kwargs["test_rate"] = 0.3
        kwargs["val_rate"] = 0.3

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pe.method_run(method, step, **kwargs)
    else:
        pe.method_run(method, step, **kwargs)


def _prepare_split(df, target_col, process_name):
    if process_name in ("dataset division", "data conversion"):
        return df, pd.DataFrame(), pd.DataFrame()

    data_train, data_test = train_test_split(df, test_size=0.3, random_state=42)
    data_train, data_val = train_test_split(data_train, test_size=0.3, random_state=42)
    return data_train, data_val, data_test


def _prepare_for_process(dataset_train, dataset_val, dataset_test, target_col, process_name):
    if process_name not in NUMERIC_REQUIRED_PROCESSES:
        return dataset_train, dataset_val, dataset_test
    if dataset_train is None or len(dataset_train) == 0:
        return dataset_train, dataset_val, dataset_test

    train = dataset_train.copy()
    val = dataset_val.copy()
    test = dataset_test.copy()

    y_train = train[target_col].copy() if target_col in train.columns else None
    y_val = val[target_col].copy() if target_col in val.columns else None
    y_test = test[target_col].copy() if target_col in test.columns else None

    X_train = train.drop(columns=[target_col], errors="ignore")
    X_val = val.drop(columns=[target_col], errors="ignore")
    X_test = test.drop(columns=[target_col], errors="ignore")

    # Use stable ordinal encoding instead of one-hot to avoid exploding feature space
    # and fit/transform feature-name mismatch across train/val/test.
    cat_cols = X_train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for col in cat_cols:
        train_vals = X_train[col].astype(str).fillna("__nan__")
        uniques = pd.Index(train_vals.unique())
        code_map = {v: i for i, v in enumerate(uniques)}
        X_train[col] = train_vals.map(code_map).astype(float)
        if col in X_val.columns:
            X_val[col] = X_val[col].astype(str).fillna("__nan__").map(code_map).fillna(-1).astype(float)
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(str).fillna("__nan__").map(code_map).fillna(-1).astype(float)

    # Force numeric and fill missing to keep downstream transformers stable.
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_val = X_val.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Align columns to training schema.
    feature_cols = X_train.columns.tolist()
    X_val = X_val.reindex(columns=feature_cols, fill_value=0.0)
    X_test = X_test.reindex(columns=feature_cols, fill_value=0.0)

    if y_train is not None:
        X_train[target_col] = y_train.values
    if y_val is not None:
        X_val[target_col] = y_val.values
    if y_test is not None:
        X_test[target_col] = y_test.values

    return X_train, X_val, X_test


def main():
    parser = argparse.ArgumentParser(description="Batch extract time coefficients for multiple datasets.")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DATASET_SHORT_NAMES.keys()),
        help="Comma-separated dataset aliases. Use --list_datasets to view aliases.",
    )
    parser.add_argument("--list_datasets", action="store_true", help="List supported dataset aliases and exit.")
    parser.add_argument("--step", type=float, default=0.1, help="Sampling step.")
    parser.add_argument("--run_num", type=int, default=1, help="Repeated runs per point for averaging.")
    parser.add_argument("--quiet", action="store_true", help="Suppress inner method output.")
    parser.add_argument("--cost_param", type=str, default="e", choices=["e", "e_dataset", "e_dataset_method"], help="Cost parameter mode.")
    parser.add_argument("--operator_space", type=str, default="legacy", choices=["legacy", "sota"], help="Operator set to profile.")
    parser.add_argument("--only_new", action="store_true", help="Profile only methods added by the selected operator space.")
    parser.add_argument(
        "--artifact_version",
        type=str,
        default=None,
        help="Write profiles to artifacts/cost_models/<version>/ without replacing current files.",
    )
    parser.add_argument(
        "--save_repeat_details",
        action="store_true",
        help="Store every repeated runtime for variability fitting.",
    )
    args = parser.parse_args()

    if args.list_datasets:
        for short_name in sorted(DATASET_SHORT_NAMES):
            print(f"{short_name} -> {DATASET_SHORT_NAMES[short_name]}")
        return

    from modules.cost_calculator.param_extractor import ParamExtractor

    project_root = find_project_root(os.path.dirname(__file__))
    if project_root is None:
        raise FileNotFoundError("Cannot locate project root.")

    dataset_inputs = [x.strip() for x in args.datasets.split(",") if x.strip()]
    dataset_names = []
    for ds in dataset_inputs:
        dataset_names.append(resolve_dataset_name(ds))

    process_methods = get_process_methods(args.operator_space, args.only_new)
    tasks = []
    for ds in dataset_names:
        for process_name, methods in process_methods.items():
            for m in methods:
                tasks.append((ds, process_name, m))

    total = len(tasks)
    done = 0

    for ds_name in dataset_names:
        _, data_path, _ = resolve_dataset_paths(project_root, ds_name)
        _, target_col = load_dataset_label(project_root, ds_name)
        df = pd.read_csv(data_path)

        for process_name, methods in process_methods.items():
            dataset_train, dataset_val, dataset_test = _prepare_split(df, target_col, process_name)
            dataset_train, dataset_val, dataset_test = _prepare_for_process(
                dataset_train, dataset_val, dataset_test, target_col, process_name
            )
            # Match real pipeline behavior: normalization runs on feature matrix only.
            if process_name == "data normalization":
                dataset_train = dataset_train.drop(columns=[target_col], errors="ignore")
                dataset_val = dataset_val.drop(columns=[target_col], errors="ignore")
                dataset_test = dataset_test.drop(columns=[target_col], errors="ignore")
            pe = ParamExtractor(dataset_train, dataset_val, dataset_test, target_col, process_name)
            non_num_features = [
                c for c in dataset_train.columns
                if c != target_col and dataset_train[c].dtype == object
            ]

            for method in methods:
                done += 1
                _progress(done, total, f"{ds_name} | {process_name} | {method}")
                _run_method(
                    pe=pe,
                    method=method,
                    step=args.step,
                    run_num=args.run_num,
                    dataset_name=ds_name,
                    process_name=process_name,
                    non_num_features=non_num_features,
                    quiet=args.quiet,
                    cost_param=args.cost_param,
                    artifact_version=args.artifact_version,
                    save_repeat_details=args.save_repeat_details or bool(args.artifact_version),
                )

    sys.stdout.write("\nDone.\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
