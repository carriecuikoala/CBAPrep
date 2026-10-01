import os
import numpy as np
import pickle
from functools import lru_cache

import pandas as pd

from modules.common.dataset_utils import (
    find_project_root,
    load_dataset_label,
    resolve_dataset_name,
    resolve_dataset_paths,
)
from modules.cost_calculator.cost_calculator import build_cost_param_payload, get_dataset_cost_params
from modules.common.operator_space import (
    get_artifact_suffix,
    get_method_options,
    get_method_order,
    normalize_operator_space,
)


METHOD_ORDER = get_method_order("legacy")

MODEL_ORDER = ["LDA", "CART", "NB", "MNB", "LR", "RF", "XGB", "MLP", "SVM"]

MODEL_ALIASES = {
    "RANDOM_FOREST": "RF",
    "RANDOMFOREST": "RF",
    "XGBOOST": "XGB",
    "XGBCLASSIFIER": "XGB",
    "SVC": "SVM",
}

STAGE_METHOD_FEATURES = [
    ("feature_encoding", "ordinal"),
    ("imputation", "MEAN"),
    ("data_normalization", "z_score"),
    ("outlier_detection", "ZSB"),
    ("feature_extraction", "pca"),
    ("feature_selection", "LC"),
]

SOTA_STAGE_METHOD_FEATURES = [
    ("duplicate_detection", "ED"),
] + STAGE_METHOD_FEATURES

DATASET_FEATURE_NAMES = [
    "log_rows",
    "log_cols",
    "log_numeric_cols",
    "log_categorical_cols",
    "numeric_ratio",
    "categorical_ratio",
    "log_cat_cardinality_sum",
    "log_cat_cardinality_max",
    "missing_rate",
    "log_rows_per_col",
    "log_class_count",
    "class_entropy",
    "majority_class_ratio",
    "minority_class_ratio",
]

MODEL_META_FEATURE_NAMES = [
    "model_is_linear",
    "model_is_tree",
    "model_is_bayes",
    "model_requires_nonnegative",
    "model_has_regularization",
    "model_generative",
]


def normalize_cost_param(cost_param):
    return str(cost_param or "e").lower()


def get_model_suffix(cost_param, operator_space="legacy", artifact_version=None):
    return get_artifact_suffix(
        normalize_cost_param(cost_param),
        operator_space,
        artifact_version=artifact_version,
    )


def normalize_benefit_model_mode(benefit_model):
    raw = str(benefit_model or "specific").strip().lower()
    if raw in {"generalized", "general", "global"}:
        return "generalized"
    if raw in {"specific", "local", "dataset_specific", "dataset-model"}:
        return "specific"
    raise ValueError("benefit_model must be 'specific' or 'generalized'.")


def normalize_downstream_model_name(model_name):
    name = str(model_name or "LDA").strip().upper().replace("-", "_").replace(" ", "_")
    return MODEL_ALIASES.get(name, name)


def dataset_feature_names():
    return DATASET_FEATURE_NAMES[:]


def model_feature_names():
    return [f"model={name}" for name in MODEL_ORDER] + MODEL_META_FEATURE_NAMES[:]


def _stage_method_features(operator_space="legacy"):
    if normalize_operator_space(operator_space) == "sota":
        return SOTA_STAGE_METHOD_FEATURES
    return STAGE_METHOD_FEATURES


def _method_feature_keys(stage_name, operator_space="legacy"):
    keys = set()
    for method in get_method_options(operator_space).get(stage_name, []):
        payload = build_cost_param_payload(
            stage_name, method, cost_param="e_dataset_method", dataset_name=None
        )
        keys.update(payload)
    excluded = {
        "mode", "method_name", "dataset_name", "stage_name",
        "dataset_rows", "dataset_cols", "e",
    }
    return sorted(key for key in keys if key not in excluded)


def _method_feature_values(stage_name, method, operator_space="legacy"):
    keys = _method_feature_keys(stage_name, operator_space)
    if method is None:
        payload = {}
    else:
        payload = build_cost_param_payload(stage_name, method, cost_param="e_dataset_method", dataset_name=None)
    return [float(payload.get(key, 0.0)) for key in keys]


def generalized_feature_names(cost_param="e", operator_space="legacy"):
    operator_space = normalize_operator_space(operator_space)
    names = [f"pipeline:{name}" for name in get_method_order(operator_space)]
    if normalize_cost_param(cost_param) == "e_dataset_method":
        for stage_name, _ in _stage_method_features(operator_space):
            for key in _method_feature_keys(stage_name, operator_space):
                names.append(f"method_param:{stage_name}_{key}")
    names.extend([f"dataset:{name}" for name in dataset_feature_names()])
    names.extend(model_feature_names())
    return names


def _safe_ratio(num, denom):
    denom = float(denom or 0.0)
    if denom <= 0:
        return 0.0
    return float(num or 0.0) / denom


@lru_cache(maxsize=128)
def _dataset_meta_features_cached(dataset_name):
    dataset_name = resolve_dataset_name(dataset_name or "google")
    params = get_dataset_cost_params(dataset_name)
    rows = float(params.get("dataset_rows", 0) or 0)
    cols = float(params.get("dataset_cols", 0) or 0)
    numeric_cols = float(params.get("numeric_cols", 0) or 0)
    categorical_cols = float(params.get("categorical_cols", 0) or 0)
    cat_cardinality_sum = float(params.get("cat_cardinality_sum", 0) or 0)
    cat_cardinality_max = float(params.get("cat_cardinality_max", 0) or 0)
    missing_rate = float(params.get("missing_rate", 0) or 0)

    class_count = 0.0
    class_entropy = 0.0
    majority_class_ratio = 0.0
    minority_class_ratio = 0.0
    try:
        _, data_path, _ = resolve_dataset_paths(project_root, dataset_name)
        _, target_col = load_dataset_label(project_root, dataset_name)
        target = pd.read_csv(data_path, usecols=[target_col])[target_col]
        counts = target.value_counts(dropna=False)
        class_count = float(len(counts))
        if len(counts) > 0:
            probs = counts.to_numpy(dtype=float) / float(counts.sum())
            class_entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(max(len(probs), 2)))
            majority_class_ratio = float(np.max(probs))
            minority_class_ratio = float(np.min(probs))
    except Exception:
        pass

    return [
        float(np.log1p(rows)),
        float(np.log1p(cols)),
        float(np.log1p(numeric_cols)),
        float(np.log1p(categorical_cols)),
        _safe_ratio(numeric_cols, cols),
        _safe_ratio(categorical_cols, cols),
        float(np.log1p(cat_cardinality_sum)),
        float(np.log1p(cat_cardinality_max)),
        missing_rate,
        float(np.log1p(_safe_ratio(rows, max(cols, 1.0)))),
        float(np.log1p(class_count)),
        class_entropy,
        majority_class_ratio,
        minority_class_ratio,
    ]


def dataset_feature_vector(dataset_name):
    return list(_dataset_meta_features_cached(resolve_dataset_name(dataset_name or "google")))


def model_feature_vector(model_name):
    model_name = normalize_downstream_model_name(model_name)
    one_hot = [1.0 if model_name == name else 0.0 for name in MODEL_ORDER]
    meta = {
        "model_is_linear": 1.0 if model_name in {"LDA", "LR", "SVM"} else 0.0,
        "model_is_tree": 1.0 if model_name in {"CART", "RF", "XGB"} else 0.0,
        "model_is_bayes": 1.0 if model_name in {"NB", "MNB"} else 0.0,
        "model_requires_nonnegative": 1.0 if model_name == "MNB" else 0.0,
        "model_has_regularization": 1.0 if model_name in {"LR", "MLP", "SVM"} else 0.0,
        "model_generative": 1.0 if model_name in {"LDA", "NB", "MNB"} else 0.0,
    }
    return one_hot + [float(meta[name]) for name in MODEL_META_FEATURE_NAMES]


def to_feature_vector(config, include_context=False, model_name=None, dataset_name=None):
    operator_space = normalize_operator_space(config.get("operator_space", "legacy"))
    method_order = get_method_order(operator_space)
    vec = [0.0 for _ in method_order]
    pairs = [
        ('method_feature_encoding', 'e_feature_encoding'),
        ('method_imputation', 'e_imputation'),
        ('method_data_normalization', 'e_data_normalization'),
        ('method_outlier_detection', 'e_outlier_detection'),
        ('method_feature_extraction', 'e_feature_extraction'),
        ('method_feature_selection', 'e_feature_selection'),
    ]
    if operator_space == "sota":
        pairs.insert(0, ('method_duplicate_detection', 'e_duplicate_detection'))
    for m_key, e_key in pairs:
        method = config.get(m_key)
        e_val = config.get(e_key, 0.0)
        if method in method_order:
            vec[method_order.index(method)] = float(e_val)
    cost_param = normalize_cost_param(config.get("cost_param", "e"))
    if cost_param == "e_dataset_method":
        stage_method_pairs = [
            ("feature_encoding", config.get("method_feature_encoding")),
            ("imputation", config.get("method_imputation")),
            ("data_normalization", config.get("method_data_normalization")),
            ("outlier_detection", config.get("method_outlier_detection")),
            ("feature_extraction", config.get("method_feature_extraction")),
            ("feature_selection", config.get("method_feature_selection")),
        ]
        if operator_space == "sota":
            stage_method_pairs.insert(0, ("duplicate_detection", config.get("method_duplicate_detection")))
        method_features = []
        for stage_name, method in stage_method_pairs:
            method_features.extend(_method_feature_values(stage_name, method, operator_space))
        vec.extend(method_features)
    if include_context:
        resolved_dataset = resolve_dataset_name(dataset_name or config.get("dataset", "google"))
        resolved_model = normalize_downstream_model_name(model_name or config.get("model", "LDA"))
        vec.extend(dataset_feature_vector(resolved_dataset))
        vec.extend(model_feature_vector(resolved_model))
    return vec


def get_pkl_store_dir(project_root):
    pkl_dir = os.path.join(project_root, "artifacts", "pkl_store")
    os.makedirs(pkl_dir, exist_ok=True)
    return pkl_dir


project_root = find_project_root(os.path.dirname(__file__))
if project_root is None:
    raise FileNotFoundError("Cannot locate project root.")
pkl_store_dir = get_pkl_store_dir(project_root)
_MODEL_CACHE = {}


def _load_model_and_scaler(
    model_name,
    dataset_name="google",
    cost_param="e",
    benefit_model="specific",
    operator_space="legacy",
    artifact_version=None,
):
    model_name = normalize_downstream_model_name(model_name)
    dataset_name = resolve_dataset_name(dataset_name or "google")
    operator_space = normalize_operator_space(operator_space)
    cost_param = get_model_suffix(cost_param, operator_space, artifact_version)
    benefit_model = normalize_benefit_model_mode(benefit_model)
    cache_key = (
        benefit_model,
        dataset_name,
        model_name,
        cost_param,
        operator_space,
        artifact_version,
    )
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if benefit_model == "generalized":
        gbr_path = os.path.join(pkl_store_dir, "gbr", f"gbr_generalized_{cost_param}.pickle")
        scaler_path = os.path.join(pkl_store_dir, "scaler", f"scaler_generalized_{cost_param}.pickle")
    else:
        gbr_path = os.path.join(
            pkl_store_dir, "gbr", f"gbr_{dataset_name}_{model_name}_{cost_param}.pickle"
        )
        scaler_path = os.path.join(
            pkl_store_dir, "scaler", f"scaler_{dataset_name}_{model_name}_{cost_param}.pickle"
        )
        if operator_space == "legacy" and not os.path.exists(gbr_path):
            gbr_path = os.path.join(
                project_root, "modules", "benefit_predictor", "gbr", f"gbr_{model_name}.pickle"
            )
        if operator_space == "legacy" and not os.path.exists(scaler_path):
            scaler_path = os.path.join(
                project_root, "modules", "benefit_predictor", "scaler", f"scaler_{model_name}.pickle"
            )
    if not os.path.exists(gbr_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Model or scaler not found for benefit_model={benefit_model}, "
            f"model={model_name}, dataset={dataset_name}, cost_param={cost_param}"
        )
    gbr_model = pickle.load(open(gbr_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))
    _MODEL_CACHE[cache_key] = (gbr_model, scaler)
    return _MODEL_CACHE[cache_key]


def acc(config, model_name=None, verbose=True, benefit_model=None):
    if model_name is None:
        model_name = config.get("model", "LDA")
    mode = normalize_benefit_model_mode(benefit_model or config.get("benefit_model", "specific"))
    pred = acc_batch([config], model_name=model_name, benefit_model=mode)[0]
    if verbose:
        print(f"Predicted accuracy ({model_name}, benefit_model={mode}): {pred}")
    return pred


def acc_batch(configs, model_name=None, benefit_model=None):
    configs = list(configs or [])
    if not configs:
        return np.asarray([], dtype=float)
    first = configs[0]
    if model_name is None:
        model_name = first.get("model", "LDA")
    dataset_name = first.get("dataset", "google")
    cost_param = normalize_cost_param(first.get("cost_param", "e"))
    operator_space = normalize_operator_space(first.get("operator_space", "legacy"))
    artifact_version = first.get("artifact_version")
    mode = normalize_benefit_model_mode(benefit_model or first.get("benefit_model", "specific"))
    gbr_model, scaler = _load_model_and_scaler(
        model_name,
        dataset_name,
        cost_param=cost_param,
        benefit_model=mode,
        operator_space=operator_space,
        artifact_version=artifact_version,
    )
    include_context = mode == "generalized"
    vectors = [
        to_feature_vector(
            config,
            include_context=include_context,
            model_name=model_name or config.get("model", "LDA"),
            dataset_name=config.get("dataset", dataset_name),
        )
        for config in configs
    ]
    vec_scaled = scaler.transform(vectors)
    pred_raw = gbr_model.predict(vec_scaled)
    pred = np.clip(pred_raw, 0, 1)
    return np.asarray(pred, dtype=float).reshape(-1)


if __name__ == "__main__":
    example_config = {
        "model": "LDA",
        "method_feature_encoding": "onehot",
        "e_feature_encoding": 0.6,
        "method_imputation": "MEAN",
        "e_imputation": 0.5,
        "method_data_normalization": "min_max",
        "e_data_normalization": 0.8,
        "method_outlier_detection": "IQR",
        "e_outlier_detection": 0.4,
        "method_feature_extraction": "pca",
        "e_feature_extraction": 0.7,
        "method_feature_selection": "Tree",
        "e_feature_selection": 0.5
    }
    acc(example_config)
