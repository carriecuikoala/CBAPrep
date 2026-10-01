import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse
import sys
import io
import contextlib
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.common.dataset_utils import (
    find_project_root as common_find_project_root,
    resolve_dataset_paths,
    load_dataset_label,
)
from modules.cost_calculator.cost_calculator import get_dataset_cost_params, build_cost_param_payload
from modules.benefit_predictor.acc_predictor import (
    get_model_suffix,
    normalize_cost_param,
    normalize_downstream_model_name,
)

from modules.methods.preprocessing.data_conversion.data_converter import DataConverter
from modules.methods.preprocessing.dataset_division.dataset_divider import DatasetDivider
from modules.methods.cleaning.nan_handling.imputer import Imputer
from modules.methods.preprocessing.data_normalization.data_normalizer import DataNormalization
from modules.methods.cleaning.outlier_detection.outlier_detector import OutlierDetector
from modules.methods.cleaning.duplicate_detection.duplicate_detector import DuplicateDetector
from modules.methods.preprocessing.feature_extraction.feature_extractor import FeatureExtractor
from modules.methods.preprocessing.feature_selection.feature_selector import FeatureSelector
from modules.ml_modules.Classification.classifier import Classifier
from modules.methods.preprocessing.feature_coding.feature_encoder import FeatureEncoder
from modules.common.operator_space import (
    get_artifact_suffix,
    get_method_options,
    normalize_operator_space,
)


def split_target(df, target_col):
    if target_col in df.columns:
        y = df[target_col].copy()
        X = df.drop(columns=[target_col])
        return X, y
    return df.copy(), pd.Series(index=df.index, dtype=object)


def attach_target(X, y, target_col):
    if y is None or y.empty:
        return X.copy()
    y = y.loc[X.index]
    return X.join(y.rename(target_col))


def sample_e(min_e=0.1, max_e=1.0):
    return float(np.around(np.random.uniform(min_e, max_e), decimals=2))


def split_benefit_training_data(dataset_train, target_col, eval_rate, random_state):
    """Create a train-internal holdout without touching final val/test data."""
    stratify = None
    if target_col in dataset_train.columns:
        target = dataset_train[target_col]
        counts = target.value_counts(dropna=False)
        if len(counts) > 1 and int(counts.min()) >= 2:
            stratify = target
    try:
        return train_test_split(
            dataset_train,
            test_size=float(eval_rate),
            random_state=int(random_state),
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            dataset_train,
            test_size=float(eval_rate),
            random_state=int(random_state),
        )


def apply_imputation(df_train, df_val, df_test, target_col, method, e):
    X_train, y_train = split_target(df_train, target_col)
    X_val, y_val = split_target(df_val, target_col)
    X_test, y_test = split_target(df_test, target_col)

    imp = Imputer(e, X_train, X_val, X_test, method=method)
    X_train, X_val, X_test, _ = imp.transform()

    df_train = attach_target(X_train, y_train, target_col)
    df_val = attach_target(X_val, y_val, target_col)
    df_test = attach_target(X_test, y_test, target_col)
    return df_train, df_val, df_test


def apply_duplicate_detection(df_train, df_val, df_test, target_col, method, e):
    X_train, y_train = split_target(df_train, target_col)
    X_val, y_val = split_target(df_val, target_col)
    X_test, y_test = split_target(df_test, target_col)
    detector = DuplicateDetector(e, X_train, X_val, X_test, method=method)
    X_train, X_val, X_test, _ = detector.transform()
    return (
        attach_target(X_train, y_train, target_col),
        attach_target(X_val, y_val, target_col),
        attach_target(X_test, y_test, target_col),
    )


def apply_feature_encoding(df_train, df_val, df_test, target_col, method, e):
    # Drop leakage-prone feature
    for df in (df_train, df_val, df_test):
        if 'Genres' in df.columns:
            df.drop(columns=['Genres'], inplace=True)

    non_num_features = [
        c for c in df_train.columns
        if c != target_col and df_train[c].dtype == object
    ]
    if not non_num_features:
        return df_train, df_val, df_test

    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    for col in non_num_features:
        df_train[col] = df_train[col].fillna('Unknown')
        df_val[col] = df_val[col].fillna('Unknown')
        df_test[col] = df_test[col].fillna('Unknown')

    dataset_fit = df_train.sample(frac=e, random_state=42)
    fe = FeatureEncoder(dataset_fit, df_train, df_val, df_test, target_col, non_num_features, method=method)
    df_train, df_val, df_test, _ = fe.transform()
    return df_train, df_val, df_test


def apply_normalization(df_train, df_val, df_test, target_col, method, e):
    X_train, y_train = split_target(df_train, target_col)
    X_val, y_val = split_target(df_val, target_col)
    X_test, y_test = split_target(df_test, target_col)

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        df_train = attach_target(X_train, y_train, target_col)
        df_val = attach_target(X_val, y_val, target_col)
        df_test = attach_target(X_test, y_test, target_col)
        return df_train, df_val, df_test

    X_train_num = X_train[numeric_cols].apply(pd.to_numeric, errors='coerce')
    X_val_num = X_val[numeric_cols].apply(pd.to_numeric, errors='coerce')
    X_test_num = X_test[numeric_cols].apply(pd.to_numeric, errors='coerce')

    fit_means = X_train_num.mean()
    X_train_num = X_train_num.fillna(fit_means)
    X_val_num = X_val_num.fillna(fit_means)
    X_test_num = X_test_num.fillna(fit_means)

    dataset_fit = X_train_num.sample(frac=e, random_state=42)
    dn = DataNormalization(dataset_fit, X_train_num, X_val_num, X_test_num)
    dn.set_params(method=method)
    X_train_scaled, X_val_scaled, X_test_scaled, _ = dn.transform()

    X_train_out = X_train.copy()
    X_val_out = X_val.copy()
    X_test_out = X_test.copy()
    X_train_out[numeric_cols] = X_train_scaled
    X_val_out[numeric_cols] = X_val_scaled
    X_test_out[numeric_cols] = X_test_scaled

    df_train = attach_target(X_train_out, y_train, target_col)
    df_val = attach_target(X_val_out, y_val, target_col)
    df_test = attach_target(X_test_out, y_test, target_col)
    return df_train, df_val, df_test


def apply_outlier_detection(df_train, df_val, df_test, target_col, method, e):
    X_train, y_train = split_target(df_train, target_col)
    X_val, y_val = split_target(df_val, target_col)
    X_test, y_test = split_target(df_test, target_col)

    od = OutlierDetector(e, X_train, X_val, X_test, method=method)
    X_train, X_val, X_test, _ = od.transform()

    df_train = attach_target(X_train, y_train, target_col)
    df_val = attach_target(X_val, y_val, target_col)
    df_test = attach_target(X_test, y_test, target_col)
    return df_train, df_val, df_test


def apply_feature_extraction(df_train, df_val, df_test, target_col, method, e):
    X_train, y_train = split_target(df_train, target_col)
    X_val, y_val = split_target(df_val, target_col)
    X_test, y_test = split_target(df_test, target_col)

    df_train_in = attach_target(X_train, y_train, target_col)
    df_val_in = attach_target(X_val, y_val, target_col)
    df_test_in = attach_target(X_test, y_test, target_col)

    fe = FeatureExtractor(e, df_train_in, df_val_in, df_test_in, target=target_col, method=method)
    df_train_out, df_val_out, df_test_out, _ = fe.transform()

    df_train = attach_target(df_train_out, y_train, target_col)
    df_val = attach_target(df_val_out, y_val, target_col)
    df_test = attach_target(df_test_out, y_test, target_col)
    return df_train, df_val, df_test


def apply_feature_selection(df_train, df_val, df_test, target_col, method, e):
    X_train, y_train = split_target(df_train, target_col)
    X_val, y_val = split_target(df_val, target_col)
    X_test, y_test = split_target(df_test, target_col)

    df_train_in = attach_target(X_train, y_train, target_col)
    df_val_in = attach_target(X_val, y_val, target_col)
    df_test_in = attach_target(X_test, y_test, target_col)

    fs = FeatureSelector(e, df_train_in, df_val_in, df_test_in, target=target_col, method=method)
    df_train_out, df_val_out, df_test_out, _ = fs.transform()

    df_train = attach_target(df_train_out, y_train, target_col)
    df_val = attach_target(df_val_out, y_val, target_col)
    df_test = attach_target(df_test_out, y_test, target_col)
    return df_train, df_val, df_test


def prepare_sampled_pipeline(
    dataset_train,
    dataset_val,
    dataset_test,
    target_col,
    methods,
    e_values,
    quiet=False,
    evaluation_split="test",
):
    """Execute one sampled preparation pipeline and return classifier-ready data."""
    started = time.perf_counter()
    df_train = dataset_train.copy()
    df_val = dataset_val.copy()
    df_test = dataset_test.copy()
    stream_ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with stream_ctx:
        if methods.get('duplicate_detection'):
            df_train, df_val, df_test = apply_duplicate_detection(
                df_train,
                df_val,
                df_test,
                target_col,
                methods['duplicate_detection'],
                e_values['duplicate_detection'],
            )
        if methods.get('feature_encoding'):
            df_train, df_val, df_test = apply_feature_encoding(
                df_train, df_val, df_test, target_col,
                methods['feature_encoding'], e_values['feature_encoding']
            )
        if methods.get('imputation'):
            df_train, df_val, df_test = apply_imputation(
                df_train, df_val, df_test, target_col,
                methods['imputation'], e_values['imputation']
            )
        if methods.get('data_normalization'):
            df_train, df_val, df_test = apply_normalization(
                df_train, df_val, df_test, target_col,
                methods['data_normalization'], e_values['data_normalization']
            )
        if methods.get('outlier_detection'):
            df_train, df_val, df_test = apply_outlier_detection(
                df_train, df_val, df_test, target_col,
                methods['outlier_detection'], e_values['outlier_detection']
            )
        if methods.get('feature_extraction'):
            df_train, df_val, df_test = apply_feature_extraction(
                df_train, df_val, df_test, target_col,
                methods['feature_extraction'], e_values['feature_extraction']
            )
        if methods.get('feature_selection'):
            df_train, df_val, df_test = apply_feature_selection(
                df_train, df_val, df_test, target_col,
                methods['feature_selection'], e_values['feature_selection']
            )

        X_train, y_train = split_target(df_train, target_col)
        if evaluation_split == "val":
            X_test, y_test = split_target(df_val, target_col)
        elif evaluation_split == "test":
            X_test, y_test = split_target(df_test, target_col)
        else:
            raise ValueError("evaluation_split must be 'val' or 'test'.")
        dataset = {
            'train': X_train.join(y_train.rename(target_col)),
            'test': X_test.join(y_test.rename(target_col)),
            'target': y_train,
            'target_test': y_test,
        }
    return dataset, time.perf_counter() - started


def score_prepared_dataset(dataset, target_col, model_name, quiet=False):
    stream_ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    started = time.perf_counter()
    with stream_ctx:
        classifier = Classifier(
            dataset,
            target=target_col,
            strategy=model_name,
            k_folds=10,
            verbose=False,
        )
        accuracy = classifier.transform()
    return accuracy, time.perf_counter() - started


def execute_sampled_pipeline(
    dataset_train,
    dataset_val,
    dataset_test,
    target_col,
    model_name,
    methods,
    e_values,
    quiet=False,
):
    dataset, prep_time = prepare_sampled_pipeline(
        dataset_train,
        dataset_val,
        dataset_test,
        target_col,
        methods,
        e_values,
        quiet=quiet,
    )
    accuracy, model_time = score_prepared_dataset(
        dataset, target_col, model_name, quiet=quiet
    )
    return accuracy, prep_time + model_time


def find_project_root(start_path):
    return common_find_project_root(start_path)


def get_pkl_store_dir(project_root):
    pkl_dir = os.path.join(project_root, "artifacts", "pkl_store")
    os.makedirs(pkl_dir, exist_ok=True)
    return pkl_dir


def atomic_pickle_dump(value, path):
    temp_path = f"{path}.tmp"
    with open(temp_path, "wb") as file:
        pickle.dump(value, file)
    os.replace(temp_path, path)


if __name__ == "__main__":
    target_col = None
    # model_names = ["LDA", "CART", "NB", "MNB", "LR"]
    model_names = ["LDA", "CART", "NB", "MNB", "LR"]
    iterations = 200

    parser = argparse.ArgumentParser(description="Generate e-accuracy dataset for downstream models.")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model name (e.g., LR/NB/CART/RF/XGB/MLP/SVM).",
    )
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names.")
    parser.add_argument("--iterations", type=int, default=None, help="Number of iterations to sample.")
    parser.add_argument("--progress_every", type=int, default=10, help="Print progress every N iterations.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-record prints.")
    parser.add_argument("--dataset", type=str, default="google", help="Dataset alias (e.g., abalone, ada, connect, jungle, run_or_walk).")
    parser.add_argument("--cost_param", type=str, default="e", choices=["e", "e_dataset", "e_dataset_method"], help="Cost parameter mode.")
    parser.add_argument("--operator_space", type=str, default="legacy", choices=["legacy", "sota"], help="Operator search space.")
    parser.add_argument("--artifact_version", type=str, default=None, help="Optional artifact version, for example sota_v2.")
    parser.add_argument("--optional_stages", action="store_true", help="Allow preprocessing stages to be skipped when valid.")
    parser.add_argument(
        "--benefit_label_source",
        default="test",
        choices=["test", "val", "train_holdout"],
        help="Data split used to generate GBDT utility labels.",
    )
    parser.add_argument("--proxy_eval_rate", type=float, default=0.25)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--max_attempt_factor", type=int, default=5, help="Maximum attempts per requested valid record.")
    parser.add_argument("--resume", action="store_true", help="Continue from partial benefit datasets when present.")
    parser.add_argument("--checkpoint_every", type=int, default=10, help="Persist partial records every N valid pipelines.")
    args = parser.parse_args()
    if args.quiet:
        warnings.filterwarnings("ignore")

    if args.models:
        model_names = [normalize_downstream_model_name(m) for m in args.models.split(",") if m.strip()]
    elif args.model:
        model_names = [normalize_downstream_model_name(args.model)]
    if args.iterations is not None:
        iterations = int(args.iterations)
    dataset_name = args.dataset.strip().lower()
    cost_param = normalize_cost_param(args.cost_param)
    operator_space = normalize_operator_space(args.operator_space)
    cost_suffix = get_artifact_suffix(
        cost_param,
        operator_space,
        artifact_version=args.artifact_version,
    )

    # 1) data conversion
    current_file_path = os.path.abspath(__file__)
    project_root = find_project_root(os.path.dirname(current_file_path))
    if project_root is None:
        raise FileNotFoundError("Cannot locate project root from script path")
    dataset_name, data_path, _ = resolve_dataset_paths(project_root, dataset_name)
    _, target_col = load_dataset_label(project_root, dataset_name)
    pkl_store_dir = get_pkl_store_dir(project_root)
    dataset_cost_params = get_dataset_cost_params(dataset_name)

    raw = pd.read_csv(data_path)
    dc = DataConverter(raw)
    data, _ = dc.transform()

    # 2) data division
    dd = DatasetDivider(data, test_rate=0.3, val_rate=0.3)
    dataset_train, dataset_val, dataset_test, _ = dd.transform()

    if args.benefit_label_source == "train_holdout":
        benefit_train, benefit_eval = split_benefit_training_data(
            dataset_train,
            target_col,
            args.proxy_eval_rate,
            args.random_state,
        )
        generation_train = benefit_train
        generation_val = benefit_eval
        generation_test = benefit_eval
        generation_eval_split = "val"
    else:
        generation_train = dataset_train
        generation_val = dataset_val
        generation_test = dataset_test
        generation_eval_split = args.benefit_label_source

    method_options = get_method_options(
        operator_space,
        include_optional=args.optional_stages,
    )
    if args.optional_stages:
        if int(dataset_cost_params.get("categorical_cols", 0) or 0) > 0:
            method_options["feature_encoding"] = [
                method for method in method_options["feature_encoding"] if method is not None
            ]
        if float(dataset_cost_params.get("missing_rate", 0.0) or 0.0) > 0:
            method_options["imputation"] = [
                method for method in method_options["imputation"] if method is not None
            ]
    duplicate_methods = method_options.get('duplicate_detection', [])
    impute_methods = method_options['imputation']
    encode_methods = method_options['feature_encoding']
    norm_methods = method_options['data_normalization']
    outlier_methods = method_options['outlier_detection']
    feat_extract_methods = method_options['feature_extraction']
    feat_select_methods = method_options['feature_selection']

    progress_every = max(1, int(args.progress_every))

    output_paths = {
        model_name: os.path.join(
            pkl_store_dir,
            f"e_accuracy_dataset_{dataset_name}_{model_name}_{cost_suffix}.pickle",
        )
        for model_name in model_names
    }
    records_by_model = {model_name: [] for model_name in model_names}
    if args.resume:
        for model_name, output_path in output_paths.items():
            if not os.path.exists(output_path):
                continue
            try:
                with open(output_path, "rb") as file:
                    records = pickle.load(file)
                if isinstance(records, list):
                    records_by_model[model_name] = records[:iterations]
            except Exception:
                records_by_model[model_name] = []
    resumed_count = min(len(records) for records in records_by_model.values())
    np.random.seed(int(args.random_state) + resumed_count)
    checkpoint_every = max(1, int(args.checkpoint_every))
    last_checkpoint = resumed_count
    failure_counts = {}
    max_attempts = max(iterations, iterations * max(1, int(args.max_attempt_factor)))
    for attempt in range(max_attempts):
        if all(len(records) >= iterations for records in records_by_model.values()):
            break
        e_impute = sample_e()
        e_duplicate = sample_e() if duplicate_methods else 0.0
        e_encode = sample_e()
        e_norm = sample_e()
        e_outlier = sample_e()
        e_feat_ext = sample_e()
        e_feat_sel = sample_e()

        encode_method = np.random.choice(encode_methods)
        impute_method = np.random.choice(impute_methods)
        norm_method = np.random.choice(norm_methods)
        outlier_method = np.random.choice(outlier_methods)
        feat_ext_method = np.random.choice(feat_extract_methods)
        feat_sel_method = np.random.choice(feat_select_methods)
        duplicate_method = np.random.choice(duplicate_methods) if duplicate_methods else None

        if duplicate_method is None:
            e_duplicate = 0.0
        if encode_method is None:
            e_encode = 0.0
        if impute_method is None:
            e_impute = 0.0
        if norm_method is None:
            e_norm = 0.0
        if outlier_method is None:
            e_outlier = 0.0
        if feat_ext_method is None:
            e_feat_ext = 0.0
        if feat_sel_method is None:
            e_feat_sel = 0.0

        methods = {
            'duplicate_detection': duplicate_method,
            'feature_encoding': encode_method,
            'imputation': impute_method,
            'data_normalization': norm_method,
            'outlier_detection': outlier_method,
            'feature_extraction': feat_ext_method,
            'feature_selection': feat_sel_method,
        }
        e_values = {
            'duplicate_detection': e_duplicate,
            'feature_encoding': e_encode,
            'imputation': e_impute,
            'data_normalization': e_norm,
            'outlier_detection': e_outlier,
            'feature_extraction': e_feat_ext,
            'feature_selection': e_feat_sel,
        }
        try:
            prepared_dataset, prep_elapsed = prepare_sampled_pipeline(
                generation_train,
                generation_val,
                generation_test,
                target_col,
                methods,
                e_values,
                quiet=args.quiet,
                evaluation_split=generation_eval_split,
            )
        except Exception as exc:
            failure_key = f"preparation/{type(exc).__name__}: {exc}"
            failure_counts[failure_key] = failure_counts.get(failure_key, 0) + 1
            if not args.quiet:
                print(f"[skip] attempt {attempt + 1}: {failure_key}")
            continue

        common_record = {
            'dataset': dataset_name,
            'cost_param': cost_param,
            'operator_space': operator_space,
            'artifact_version': args.artifact_version,
            'benefit_label_source': args.benefit_label_source,
            'proxy_eval_rate': args.proxy_eval_rate if args.benefit_label_source == "train_holdout" else None,
            'benefit_fit_rows': int(len(generation_train)),
            'benefit_eval_rows': int(len(generation_val if generation_eval_split == "val" else generation_test)),
            'method_feature_encoding': encode_method,
            'e_feature_encoding': e_encode,
            'method_imputation': impute_method,
            'e_imputation': e_impute,
            'method_data_normalization': norm_method,
            'e_data_normalization': e_norm,
            'method_outlier_detection': outlier_method,
            'e_outlier_detection': e_outlier,
            'method_feature_extraction': feat_ext_method,
            'e_feature_extraction': e_feat_ext,
            'method_feature_selection': feat_sel_method,
            'e_feature_selection': e_feat_sel,
        }
        if duplicate_method:
            common_record['method_duplicate_detection'] = duplicate_method
            common_record['e_duplicate_detection'] = e_duplicate
        if cost_param in ("e_dataset", "e_dataset_method"):
            common_record.update(dataset_cost_params)
        if cost_param == "e_dataset_method":
            stage_method_pairs = [
                ("feature_encoding", encode_method),
                ("imputation", impute_method),
                ("data_normalization", norm_method),
                ("outlier_detection", outlier_method),
                ("feature_extraction", feat_ext_method),
                ("feature_selection", feat_sel_method),
            ]
            if duplicate_method:
                stage_method_pairs.insert(0, ("duplicate_detection", duplicate_method))
            for stage_name, method_name in stage_method_pairs:
                payload = build_cost_param_payload(
                    stage_name,
                    method_name,
                    cost_param="e_dataset_method",
                    dataset_name=None,
                )
                for key, value in payload.items():
                    if key in (
                        "mode", "method_name", "dataset_name", "stage_name",
                        "dataset_rows", "dataset_cols", "e",
                    ):
                        continue
                    common_record[f"{stage_name}_{key}"] = value

        pending_models = [
            model_name
            for model_name in model_names
            if len(records_by_model[model_name]) < iterations
        ]
        model_results = {}
        slow_models = [model for model in pending_models if model in {"MLP", "SVM"}]
        fast_models = [model for model in pending_models if model not in {"MLP", "SVM"}]

        for model_name in fast_models:
            try:
                model_results[model_name] = score_prepared_dataset(
                    prepared_dataset, target_col, model_name, quiet=args.quiet
                )
            except Exception as exc:
                model_results[model_name] = exc

        if len(slow_models) > 1:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(
                        score_prepared_dataset,
                        prepared_dataset,
                        target_col,
                        model_name,
                        False,
                    ): model_name
                    for model_name in slow_models
                }
                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        model_results[model_name] = future.result()
                    except Exception as exc:
                        model_results[model_name] = exc
        elif slow_models:
            model_name = slow_models[0]
            try:
                model_results[model_name] = score_prepared_dataset(
                    prepared_dataset, target_col, model_name, quiet=args.quiet
                )
            except Exception as exc:
                model_results[model_name] = exc

        for model_name in pending_models:
            result = model_results[model_name]
            if isinstance(result, Exception):
                failure_key = f"{model_name}/{type(result).__name__}: {result}"
                failure_counts[failure_key] = failure_counts.get(failure_key, 0) + 1
                if not args.quiet:
                    print(f"[skip] attempt {attempt + 1}: {failure_key}")
                continue
            accuracy, model_elapsed = result
            if accuracy is None or not np.isfinite(float(accuracy)):
                failure_key = f"{model_name}/invalid_accuracy: {accuracy}"
                failure_counts[failure_key] = failure_counts.get(failure_key, 0) + 1
                if not args.quiet:
                    print(f"[skip] attempt {attempt + 1}: {failure_key}")
                continue
            model_records = records_by_model[model_name]
            record = dict(common_record)
            record['accuracy'] = accuracy
            record['elapsed'] = prep_elapsed + model_elapsed
            model_records.append(record)
            if not args.quiet:
                print(f"[{model_name}] generated record:", record)
            completed = len(model_records)
            if completed % progress_every == 0 or completed == iterations:
                sys.stdout.write(f"[run] {model_name} {completed}/{iterations}\n")
                sys.stdout.flush()

        min_completed = min(len(records) for records in records_by_model.values())
        if min_completed >= last_checkpoint + checkpoint_every:
            for model_name, records in records_by_model.items():
                atomic_pickle_dump(records, output_paths[model_name])
            last_checkpoint = min_completed

    for model_name, e_accuracy_dataset in records_by_model.items():
        if len(e_accuracy_dataset) < iterations:
            summary = sorted(failure_counts.items(), key=lambda item: item[1], reverse=True)[:5]
            raise RuntimeError(
                f"Generated {len(e_accuracy_dataset)}/{iterations} valid records for {model_name} after "
                f"{max_attempts} attempts. Frequent failures: {summary}"
            )
        output_path = output_paths[model_name]
        atomic_pickle_dump(e_accuracy_dataset, output_path)
        print(f"saved: {output_path}")
