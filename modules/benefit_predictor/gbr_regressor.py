import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from modules.benefit_predictor.acc_predictor import (
    generalized_feature_names,
    get_model_suffix,
    normalize_benefit_model_mode,
    normalize_cost_param,
    normalize_downstream_model_name,
    to_feature_vector,
)
from modules.common.dataset_utils import find_project_root, resolve_dataset_name
from modules.common.operator_space import normalize_operator_space


def get_pkl_store_dir(project_root):
    pkl_dir = os.path.join(project_root, "artifacts", "pkl_store")
    os.makedirs(pkl_dir, exist_ok=True)
    return pkl_dir


def parse_csv_list(text, cast=str):
    return [cast(x.strip()) for x in str(text).split(",") if x.strip()]


def resolve_dataset_list(dataset, datasets):
    if datasets:
        values = parse_csv_list(datasets, resolve_dataset_name)
    else:
        values = parse_csv_list(dataset, resolve_dataset_name)
    return values or ["google"]


def resolve_model_list(model, models):
    if models:
        values = parse_csv_list(models, normalize_downstream_model_name)
    else:
        values = parse_csv_list(model, normalize_downstream_model_name)
    return values or ["LDA"]


def find_benefit_dataset_path(
    project_root,
    dataset_name,
    model_name,
    cost_param,
    operator_space="legacy",
    artifact_version=None,
):
    pkl_store_dir = get_pkl_store_dir(project_root)
    operator_space = normalize_operator_space(operator_space)
    suffix = get_model_suffix(cost_param, operator_space, artifact_version)
    candidates = [
        os.path.join(pkl_store_dir, f"e_accuracy_dataset_{dataset_name}_{model_name}_{suffix}.pickle"),
        os.path.join(
            project_root,
            "modules",
            "benefit_predictor",
            "predict_data_generation",
            f"e_accuracy_dataset_{dataset_name}_{model_name}_{suffix}.pickle",
        ),
    ]
    if operator_space == "legacy":
        candidates.extend([
            os.path.join(pkl_store_dir, f"e_accuracy_dataset_{dataset_name}_{model_name}.pickle"),
            os.path.join(
                project_root,
                "modules",
                "benefit_predictor",
                "predict_data_generation",
                f"e_accuracy_dataset_{dataset_name}_{model_name}.pickle",
            ),
            os.path.join(
                project_root,
                "modules",
                "benefit_predictor",
                "predict_data_generation",
                f"e_accuracy_dataset_{model_name}.pickle",
            ),
        ])
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_records(
    project_root,
    dataset_name,
    model_name,
    cost_param,
    operator_space="legacy",
    artifact_version=None,
):
    operator_space = normalize_operator_space(operator_space)
    source_path = find_benefit_dataset_path(
        project_root,
        dataset_name,
        model_name,
        cost_param,
        operator_space,
        artifact_version,
    )
    if not source_path:
        raise FileNotFoundError(
            f"Missing e_accuracy dataset for dataset={dataset_name}, model={model_name}, cost_param={cost_param}."
        )
    records = pickle.load(open(source_path, "rb"))
    out = []
    for record in records:
        item = dict(record)
        item["dataset"] = resolve_dataset_name(item.get("dataset", dataset_name))
        item["model"] = normalize_downstream_model_name(item.get("model", model_name))
        item["cost_param"] = normalize_cost_param(item.get("cost_param", cost_param))
        item["operator_space"] = normalize_operator_space(
            item.get("operator_space", operator_space)
        )
        out.append(item)
    return out, source_path


def build_training_frame(records, include_context, default_model=None, default_dataset=None):
    X = [
        to_feature_vector(
            record,
            include_context=include_context,
            model_name=record.get("model", default_model),
            dataset_name=record.get("dataset", default_dataset),
        )
        for record in records
    ]
    y = pd.Series([record.get("accuracy", None) for record in records], name="accuracy")
    mask = ~pd.isnull(y)
    X_clean = [x for x, keep in zip(X, mask) if keep]
    y_clean = y[mask].reset_index(drop=True)
    if not X_clean:
        raise ValueError("No valid accuracy labels were found in training records.")
    return X_clean, y_clean


def train_gbr(X, y, args):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=int(args.random_state),
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_test)}")
    print(f"Feature dimension: {len(X_train[0]) if X_train else 0}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    gbr_model = GradientBoostingRegressor(
        n_estimators=int(args.gbr_estimators),
        learning_rate=float(args.gbr_lr),
        max_depth=int(args.gbr_depth),
        random_state=int(args.random_state),
    )
    gbr_model.fit(X_train_scaled, y_train)
    y_pred = np.clip(gbr_model.predict(X_test_scaled), 0, 1)
    mse = mean_squared_error(y_test, y_pred)
    print(f"GBR MSE: {mse}")
    return gbr_model, scaler, mse


def save_model(project_root, gbr_model, scaler, gbr_name, scaler_name, metadata=None):
    pkl_store_dir = get_pkl_store_dir(project_root)
    gbr_dir = os.path.join(pkl_store_dir, "gbr")
    scaler_dir = os.path.join(pkl_store_dir, "scaler")
    os.makedirs(gbr_dir, exist_ok=True)
    os.makedirs(scaler_dir, exist_ok=True)

    gbr_path = os.path.join(gbr_dir, gbr_name)
    scaler_path = os.path.join(scaler_dir, scaler_name)
    pickle.dump(gbr_model, open(gbr_path, "wb"))
    pickle.dump(scaler, open(scaler_path, "wb"))
    print(f"Saved GBR: {gbr_path}")
    print(f"Saved scaler: {scaler_path}")

    if metadata:
        metadata_path = os.path.join(gbr_dir, gbr_name.replace(".pickle", "_metadata.pickle"))
        pickle.dump(metadata, open(metadata_path, "wb"))
        print(f"Saved metadata: {metadata_path}")
    return gbr_path, scaler_path


def train_specific(project_root, args, dataset_name, model_name, cost_param):
    operator_space = normalize_operator_space(args.operator_space)
    records, source_path = load_records(
        project_root,
        dataset_name,
        model_name,
        cost_param,
        operator_space,
        args.artifact_version,
    )
    X, y = build_training_frame(
        records,
        include_context=False,
        default_model=model_name,
        default_dataset=dataset_name,
    )
    print(f"Source: {source_path}")
    gbr_model, scaler, mse = train_gbr(X, y, args)
    suffix = get_model_suffix(cost_param, operator_space, args.artifact_version)
    metadata = {
        "benefit_model": "specific",
        "dataset": dataset_name,
        "model": model_name,
        "cost_param": cost_param,
        "operator_space": operator_space,
        "artifact_version": args.artifact_version,
        "source_path": source_path,
        "samples": int(len(X)),
        "feature_dim": int(len(X[0])),
        "mse": float(mse),
    }
    return save_model(
        project_root,
        gbr_model,
        scaler,
        f"gbr_{dataset_name}_{model_name}_{suffix}.pickle",
        f"scaler_{dataset_name}_{model_name}_{suffix}.pickle",
        metadata=metadata,
    )


def train_generalized(project_root, args, dataset_list, model_list, cost_param):
    operator_space = normalize_operator_space(args.operator_space)
    all_records = []
    sources = []
    for dataset_name in dataset_list:
        for model_name in model_list:
            try:
                records, source_path = load_records(
                    project_root,
                    dataset_name,
                    model_name,
                    cost_param,
                    operator_space,
                    args.artifact_version,
                )
            except FileNotFoundError as exc:
                print(f"[warn] {exc}", file=sys.stderr)
                continue
            all_records.extend(records)
            sources.append(source_path)
            print(f"Loaded {len(records)} records: dataset={dataset_name}, model={model_name}")

    if not all_records:
        raise FileNotFoundError("No e_accuracy training records were found for generalized GBR training.")

    X, y = build_training_frame(all_records, include_context=True)
    gbr_model, scaler, mse = train_gbr(X, y, args)
    suffix = get_model_suffix(cost_param, operator_space, args.artifact_version)
    metadata = {
        "benefit_model": "generalized",
        "datasets": dataset_list,
        "models": model_list,
        "cost_param": cost_param,
        "operator_space": operator_space,
        "artifact_version": args.artifact_version,
        "source_paths": sources,
        "samples": int(len(X)),
        "feature_dim": int(len(X[0])),
        "feature_names": generalized_feature_names(cost_param, operator_space),
        "mse": float(mse),
    }
    return save_model(
        project_root,
        gbr_model,
        scaler,
        f"gbr_generalized_{suffix}.pickle",
        f"scaler_generalized_{suffix}.pickle",
        metadata=metadata,
    )


def main():
    parser = argparse.ArgumentParser(description="Train GBR regressor for accuracy prediction.")
    parser.add_argument(
        "--model",
        type=str,
        default="LDA",
        help="Downstream model name (e.g., LR/NB/CART/RF/XGB/MLP/SVM).",
    )
    parser.add_argument("--models", type=str, default=None, help="Comma-separated downstream model names for generalized training.")
    parser.add_argument("--dataset", type=str, default="google", help="Dataset alias (e.g., abalone, ada, connect, jungle, run_or_walk).")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset aliases for generalized training.")
    parser.add_argument("--cost_param", type=str, default="e", choices=["e", "e_dataset", "e_dataset_method"], help="Cost parameter mode.")
    parser.add_argument("--benefit_model", type=str, default="specific", choices=["specific", "generalized"], help="GBDT benefit predictor type.")
    parser.add_argument("--gbr_estimators", type=int, default=100, help="Number of boosting stages for GBR.")
    parser.add_argument("--gbr_lr", type=float, default=0.1, help="Learning rate for GBR.")
    parser.add_argument("--gbr_depth", type=int, default=3, help="Max depth of individual regression trees in GBR.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for splitting and GBR.")
    parser.add_argument("--operator_space", type=str, default="legacy", choices=["legacy", "sota"], help="Operator search space.")
    parser.add_argument(
        "--artifact_version",
        type=str,
        default=None,
        help="Optional artifact version suffix, for example sota_v2.",
    )
    args = parser.parse_args()

    project_root = find_project_root(os.path.dirname(__file__))
    if project_root is None:
        raise FileNotFoundError("Cannot locate project root.")

    cost_param = normalize_cost_param(args.cost_param)
    benefit_model = normalize_benefit_model_mode(args.benefit_model)
    dataset_list = resolve_dataset_list(args.dataset, args.datasets)
    model_list = resolve_model_list(args.model, args.models)

    if benefit_model == "specific":
        train_specific(project_root, args, dataset_list[0], model_list[0], cost_param)
    else:
        train_generalized(project_root, args, dataset_list, model_list, cost_param)


if __name__ == "__main__":
    main()
