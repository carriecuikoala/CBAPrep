"""
cost_calculator.py - 成本计算模块

该模块提供机器任务和人工任务的时间与成本计算函数
所有函数设计为向量化运算，支持numpy数组输入
"""

import numpy as np
import os
import pickle
import pandas as pd
from functools import lru_cache
from .human_cost import HumanCostCalculator as Hcc
from .machine_cost import  MachineCostCalculator as Mcc
from modules.common.dataset_utils import resolve_dataset_name
from modules.common.dataset_utils import find_project_root, resolve_dataset_paths, load_dataset_label


class Cost():
    def machine_time(coeff,e_machine):
        """
        计算机器任务的时间消耗

        参数:
        e (float/ndarray): 执行程度，范围[0,1]
        coeff (list): 时间系数 [线性项系数, 二次项系数]

        返回:
        float/ndarray: 计算的时间成本

        """
        mcc = Mcc()
        return max(0.0, mcc.machine_time(coeff,e_machine))


    def machine_cost(coeff):
        """
        计算机器任务的金钱消耗

        参数:
        x (float/ndarray): 执行程度，范围[0,1]
        time:机器任务执行时间
        coeff (list): 成本系数 [基础成本系数, 指数系数]

        返回:
        float/ndarray: 计算的金钱成本

        示例:
        >>> machine_cost(0.3, [100, 0.5])
        100 * e^(0.5*0.3) ≈ 100*1.1618 ≈ 116.18
        """
        mcc = Mcc()
        return mcc.machine_cost(coeff)


    def human_time(task_type, d, d_max, n, n_max, mu ,e,number):
        """
        计算人工任务的时间消耗（调用HumanCostCalculator）

        参数:
        task_type (str): 任务类型
        d (int): 数据维度
        d_max (int): 最大数据维度
        n (int): 样本量
        n_max (int): 最大样本量
        mu (float): 模糊度 (0-1)
        e：任务处理程度
        number：人员数量

        返回:
        float: 计算的时间成本
        """
        hcc = Hcc()
        # 假设calculate_task_cost返回 (cost_calculator, time)
        cost, time = hcc.human_time_and_cost(task_type, d, d_max, n, n_max, mu,e,number)
        return time

    def human_cost(task_type, d, d_max, n, n_max, mu,e,number):
        """
        计算人工任务的金钱消耗（调用HumanCostCalculator）

        参数:
        task_type (str): 任务类型
        d (int): 数据维度
        d_max (int): 最大数据维度
        n (int): 样本量
        n_max (int): 最大样本量
        mu (float): 模糊度 (0-1)

        返回:
        float: 计算的金钱成本
        """
        hcc = Hcc()
        cost, time = hcc.human_time_and_cost(task_type, d, d_max, n, n_max, mu,e,number)
        return cost


def _find_project_root(start_path):
    cur = os.path.abspath(start_path)
    while True:
        candidate = os.path.join(cur, 'data')
        if os.path.isdir(candidate):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def _get_pkl_store_dir(base_dir=None):
    if base_dir:
        pkl_dir = os.path.join(base_dir, "artifacts", "pkl_store")
        os.makedirs(pkl_dir, exist_ok=True)
        return pkl_dir
    project_root = _find_project_root(os.path.dirname(__file__))
    if project_root:
        pkl_dir = os.path.join(project_root, "artifacts", "pkl_store")
        os.makedirs(pkl_dir, exist_ok=True)
        return pkl_dir
    return None


def _find_time_file(method, step, base_dir=None, dataset=None, cost_param="e", strict_cost_param=False):
    dataset = resolve_dataset_name(dataset) if dataset else None
    cost_param = str(cost_param or "e").lower()
    suffix = f"time_{method}_step{int(step * 100)}.pkl"
    alt_name = f"{step}_{method}_time.pkl"
    full_name = f"time_{method}_full.pkl"
    ds_suffix = f"time_{dataset}_{method}_step{int(step * 100)}.pkl" if dataset else None
    ds_alt_name = f"{dataset}_{step}_{method}_time.pkl" if dataset else None
    ds_full_name = f"time_{dataset}_{method}_full.pkl" if dataset else None
    cp_suffix = f"time_{method}_step{int(step * 100)}_{cost_param}.pkl"
    cp_full_name = f"time_{method}_full_{cost_param}.pkl"
    ds_cp_suffix = f"time_{dataset}_{method}_step{int(step * 100)}_{cost_param}.pkl" if dataset else None
    ds_cp_full_name = f"time_{dataset}_{method}_full_{cost_param}.pkl" if dataset else None
    search_roots = []
    if base_dir:
        search_roots.append(base_dir)
    pkl_store_dir = _get_pkl_store_dir(base_dir=base_dir)
    if pkl_store_dir:
        search_roots.append(pkl_store_dir)
    project_root = _find_project_root(os.path.dirname(__file__))
    if project_root:
        search_roots.append(project_root)
        search_roots.append(os.path.join(project_root, "modules", "cost_calculator"))
    search_roots.append(os.getcwd())

    for root in search_roots:
        if root:
            if dataset and ds_cp_suffix and os.path.exists(os.path.join(root, ds_cp_suffix)):
                return os.path.join(root, ds_cp_suffix)
            if dataset and ds_cp_full_name and os.path.exists(os.path.join(root, ds_cp_full_name)):
                return os.path.join(root, ds_cp_full_name)
            if strict_cost_param and cost_param != "e":
                continue
            if dataset and ds_suffix and os.path.exists(os.path.join(root, ds_suffix)):
                return os.path.join(root, ds_suffix)
            if dataset and ds_alt_name and os.path.exists(os.path.join(root, ds_alt_name)):
                return os.path.join(root, ds_alt_name)
            if dataset and ds_full_name and os.path.exists(os.path.join(root, ds_full_name)):
                return os.path.join(root, ds_full_name)
            if os.path.exists(os.path.join(root, cp_suffix)):
                return os.path.join(root, cp_suffix)
            if os.path.exists(os.path.join(root, cp_full_name)):
                return os.path.join(root, cp_full_name)
            if os.path.exists(os.path.join(root, suffix)):
                return os.path.join(root, suffix)
            if os.path.exists(os.path.join(root, alt_name)):
                return os.path.join(root, alt_name)
            if os.path.exists(os.path.join(root, full_name)):
                return os.path.join(root, full_name)

    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            if dataset and ds_cp_suffix in filenames:
                return os.path.join(dirpath, ds_cp_suffix)
            if dataset and ds_cp_full_name in filenames:
                return os.path.join(dirpath, ds_cp_full_name)
            if strict_cost_param and cost_param != "e":
                continue
            if dataset and ds_suffix in filenames:
                return os.path.join(dirpath, ds_suffix)
            if dataset and ds_alt_name in filenames:
                return os.path.join(dirpath, ds_alt_name)
            if dataset and ds_full_name in filenames:
                return os.path.join(dirpath, ds_full_name)
            if cp_suffix in filenames:
                return os.path.join(dirpath, cp_suffix)
            if cp_full_name in filenames:
                return os.path.join(dirpath, cp_full_name)
            if suffix in filenames:
                return os.path.join(dirpath, suffix)
            if alt_name in filenames:
                return os.path.join(dirpath, alt_name)
            if full_name in filenames:
                return os.path.join(dirpath, full_name)
    return None


def _fit_time_coeff(time_list, step):
    if not time_list:
        return [0.0, 0.0]
    if len(time_list) < 3:
        avg_time = float(np.mean(time_list))
        return [0.0, max(0.0, avg_time)]
    x = np.linspace(step, 1, len(time_list))
    y = np.array(time_list)
    k, b = np.polyfit(x, y, 1)
    # prevent negative intercept causing negative time
    if b < 0:
        b = 0.0
    return [float(k), float(b)]


PROFILE_PROCESS_NAMES = {
    "duplicate_detection": "duplicate detection",
    "feature_encoding": "feature coding",
    "imputation": "imputation",
    "data_normalization": "data normalization",
    "outlier_detection": "outlier detection",
    "feature_extraction": "feature extraction",
    "feature_selection": "feature selection",
    "data_conversion": "data conversion",
    "dataset_division": "dataset division",
}


@lru_cache(maxsize=16)
def _load_versioned_runtime_models_cached(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_versioned_runtime_models(base_dir=None, version=None):
    if not version:
        return None
    project_root = base_dir or _find_project_root(os.path.dirname(__file__))
    if not project_root:
        return None
    path = os.path.join(
        project_root,
        "artifacts",
        "cost_models",
        str(version),
        "runtime_cost_models.pkl",
    )
    return _load_versioned_runtime_models_cached(path)


def get_versioned_runtime_coeff(
    dataset,
    stage_name,
    method,
    cost_param="e_dataset_method",
    runtime_mode="mean",
    base_dir=None,
    version=None,
):
    payload = load_versioned_runtime_models(base_dir=base_dir, version=version)
    if not payload:
        raise FileNotFoundError(
            f"Versioned runtime model not found: artifacts/cost_models/{version}/runtime_cost_models.pkl"
        )
    dataset = resolve_dataset_name(dataset) if dataset else ""
    process_name = PROFILE_PROCESS_NAMES.get(str(stage_name), str(stage_name))
    key = "|".join([dataset, process_name, str(method), str(cost_param or "e")])
    model = payload.get("models", {}).get(key)
    if not model:
        raise KeyError(
            f"Runtime coefficient missing for dataset={dataset}, stage={stage_name}, method={method}, "
            f"cost_param={cost_param}, version={version}."
        )
    runtime_mode = str(runtime_mode or "mean").lower()
    if runtime_mode == "mean":
        return list(model["mean_coeff"]), model
    if runtime_mode in {"q90", "quantile", "risk"}:
        return list(model["quantile_coeff"]), model
    raise ValueError("runtime_mode must be 'mean' or 'q90'.")


METHOD_PARAM_TEMPLATES = {
    "duplicate_detection": {
        "ED": {"approximate": 0, "string_similarity": 0},
        "AD": {"approximate": 1, "string_similarity": 1},
    },
    "feature_encoding": {
        "ordinal": {"encoding_dim_factor": 1, "needs_target": 0},
        "onehot": {"encoding_dim_factor": 2, "needs_target": 0},
        "count": {"encoding_dim_factor": 1, "needs_target": 0},
        "target": {"encoding_dim_factor": 1, "needs_target": 1},
    },
    "imputation": {
        "DROP": {"iterative": 0, "neighbor_based": 0},
        "MEAN": {"iterative": 0, "neighbor_based": 0},
        "MEDIAN": {"iterative": 0, "neighbor_based": 0},
        "MF": {"iterative": 1, "neighbor_based": 0},
        "MICE": {"iterative": 1, "neighbor_based": 0},
        "KNN": {"iterative": 0, "neighbor_based": 1},
        "EM": {"iterative": 1, "neighbor_based": 0},
        "DT": {"iterative": 1, "neighbor_based": 0},
        "DUMMY": {"iterative": 0, "neighbor_based": 0},
    },
    "data_normalization": {
        "z_score": {"robust_scale": 0, "nonlinear_map": 0},
        "robust": {"robust_scale": 1, "nonlinear_map": 0},
        "min_max": {"robust_scale": 0, "nonlinear_map": 0},
        "max_abs": {"robust_scale": 0, "nonlinear_map": 0},
        "normalizer": {"robust_scale": 0, "nonlinear_map": 0},
        "sigmoid": {"robust_scale": 0, "nonlinear_map": 1},
        "quantile_uniform": {"robust_scale": 1, "nonlinear_map": 1},
        "quantile_normal": {"robust_scale": 1, "nonlinear_map": 1},
        "power": {"robust_scale": 0, "nonlinear_map": 1},
        "kbins_uniform": {"robust_scale": 0, "nonlinear_map": 1},
        "kbins_quantile": {"robust_scale": 1, "nonlinear_map": 1},
    },
    "outlier_detection": {
        "ZSB": {"density_based": 0, "global_rule": 1},
        "IQR": {"density_based": 0, "global_rule": 1},
        "LOF": {"density_based": 1, "global_rule": 0},
        "MAD": {"density_based": 0, "global_rule": 1},
    },
    "feature_extraction": {
        "pca": {"supervised": 0, "projection_based": 1},
        "truncated_svd": {"supervised": 0, "projection_based": 1},
        "cfs": {"supervised": 1, "projection_based": 0},
        "polynomial": {"supervised": 0, "projection_based": 0, "tree_based": 0},
        "interaction": {"supervised": 0, "projection_based": 0, "tree_based": 0},
        "incremental_pca": {"supervised": 0, "projection_based": 1, "tree_based": 0},
        "kernel_pca": {"supervised": 0, "projection_based": 1, "tree_based": 0},
        "random_trees_embedding": {"supervised": 0, "projection_based": 0, "tree_based": 1},
        "pca_arpack": {"supervised": 0, "projection_based": 1, "tree_based": 0},
        "pca_randomized": {"supervised": 0, "projection_based": 1, "tree_based": 0},
    },
    "feature_selection": {
        "LC": {"tree_based": 0, "model_based": 0, "filter_based": 1},
        "VAR": {"tree_based": 0, "model_based": 0, "filter_based": 1},
        "Tree": {"tree_based": 1, "model_based": 1, "filter_based": 0},
        "SVC": {"tree_based": 0, "model_based": 1, "filter_based": 0},
        "MR": {"tree_based": 0, "model_based": 0, "filter_based": 1},
        "WR": {"tree_based": 0, "model_based": 0, "filter_based": 1},
    },
}


@lru_cache(maxsize=64)
def get_dataset_cost_params(dataset_name=None):
    dataset_name = resolve_dataset_name(dataset_name) if dataset_name else None
    if not dataset_name:
        return {}

    project_root = find_project_root(os.path.dirname(__file__))
    if project_root is None:
        return {}

    try:
        _, data_path, _ = resolve_dataset_paths(project_root, dataset_name)
        _, label = load_dataset_label(project_root, dataset_name)
        df = pd.read_csv(data_path)
    except Exception:
        return {}

    feature_cols = [c for c in df.columns if c != label]
    feature_df = df[feature_cols] if feature_cols else pd.DataFrame()
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    cat_cardinalities = [
        int(feature_df[c].nunique(dropna=True))
        for c in categorical_cols
        if c in feature_df.columns
    ]
    missing_rate = float(feature_df.isna().mean().mean()) if feature_cols else 0.0
    return {
        "dataset_rows": int(df.shape[0]),
        "dataset_cols": int(len(feature_cols)),
        "numeric_cols": int(len(numeric_cols)),
        "categorical_cols": int(len(categorical_cols)),
        "cat_cardinality_sum": int(sum(cat_cardinalities)),
        "cat_cardinality_max": int(max(cat_cardinalities, default=0)),
        "missing_rate": missing_rate,
    }


def get_method_cost_params(stage_name, method):
    stage_name = str(stage_name or "")
    method = str(method or "")
    params = METHOD_PARAM_TEMPLATES.get(stage_name, {}).get(method, {})
    out = {"method_name": method}
    out.update(params)
    return out


def build_cost_param_payload(stage_name, method, cost_param="e", dataset_name=None):
    cost_param = str(cost_param or "e").lower()
    resolved_dataset = resolve_dataset_name(dataset_name) if dataset_name else None
    payload = {
        "mode": cost_param,
        "e": 1,
        "stage_name": str(stage_name or ""),
        "dataset_name": resolved_dataset or "",
    }
    if cost_param in ("e_dataset", "e_dataset_method"):
        payload.update(get_dataset_cost_params(resolved_dataset))
    if cost_param == "e_dataset_method":
        payload.update(get_method_cost_params(stage_name, method))
    return payload


def compute_cost_param_multiplier(cost_param_payload=None):
    payload = dict(cost_param_payload or {})
    mode = str(payload.get("mode", "e")).lower()
    if mode == "e":
        return 1.0

    multiplier = 1.0

    if mode in ("e_dataset", "e_dataset_method"):
        rows = max(float(payload.get("dataset_rows", 0) or 0), 1.0)
        cols = max(float(payload.get("dataset_cols", 0) or 0), 1.0)
        row_term = np.clip(np.log10(rows) - 4.0, -2.0, 2.0)
        col_term = np.clip(np.log10(cols) - 1.5, -2.0, 2.0)
        multiplier *= (1.0 + 0.08 * row_term)
        multiplier *= (1.0 + 0.05 * col_term)

    if mode == "e_dataset_method":
        method_bonus = 0.0
        method_bonus += 0.06 * max(float(payload.get("encoding_dim_factor", 1)) - 1.0, 0.0)
        method_bonus += 0.03 * float(payload.get("needs_target", 0))
        method_bonus += 0.08 * float(payload.get("iterative", 0))
        method_bonus += 0.06 * float(payload.get("neighbor_based", 0))
        method_bonus += 0.02 * float(payload.get("robust_scale", 0))
        method_bonus += 0.03 * float(payload.get("nonlinear_map", 0))
        method_bonus += 0.06 * float(payload.get("density_based", 0))
        method_bonus += 0.02 * float(payload.get("global_rule", 0))
        method_bonus += 0.04 * float(payload.get("projection_based", 0))
        method_bonus += 0.03 * float(payload.get("supervised", 0))
        method_bonus += 0.05 * float(payload.get("tree_based", 0))
        method_bonus += 0.04 * float(payload.get("model_based", 0))
        method_bonus += 0.01 * float(payload.get("filter_based", 0))
        method_bonus += 0.08 * float(payload.get("approximate", 0))
        method_bonus += 0.04 * float(payload.get("string_similarity", 0))
        multiplier *= (1.0 + method_bonus)

    return max(0.1, float(multiplier))


def _calibrator_path(base_dir=None, cost_param="e_dataset_method"):
    pkl_dir = _get_pkl_store_dir(base_dir=base_dir)
    if not pkl_dir:
        return None
    return os.path.join(pkl_dir, f"machine_cost_calibrator_{str(cost_param or 'e').lower()}.pkl")


@lru_cache(maxsize=16)
def _load_machine_cost_calibrator_cached(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def clear_machine_cost_calibrator_cache():
    _load_machine_cost_calibrator_cached.cache_clear()


def load_machine_cost_calibrator(base_dir=None, cost_param="e_dataset_method"):
    path = _calibrator_path(base_dir=base_dir, cost_param=cost_param)
    return _load_machine_cost_calibrator_cached(path)


def build_machine_cost_calibration_features(cost_param_payload=None):
    payload = dict(cost_param_payload or {})
    rows = max(float(payload.get("dataset_rows", 0) or 0), 0.0)
    cols = max(float(payload.get("dataset_cols", 0) or 0), 0.0)
    numeric_cols = max(float(payload.get("numeric_cols", 0) or 0), 0.0)
    categorical_cols = max(float(payload.get("categorical_cols", 0) or 0), 0.0)
    cat_cardinality_sum = max(float(payload.get("cat_cardinality_sum", 0) or 0), 0.0)
    cat_cardinality_max = max(float(payload.get("cat_cardinality_max", 0) or 0), 0.0)
    missing_rate = max(float(payload.get("missing_rate", 0) or 0), 0.0)
    safe_cols = max(cols, 1.0)

    features = {
        "intercept": 1.0,
        "log_rows": float(np.log1p(rows)),
        "log_cols": float(np.log1p(cols)),
        "numeric_ratio": float(numeric_cols / safe_cols),
        "categorical_ratio": float(categorical_cols / safe_cols),
        "log_cat_cardinality_sum": float(np.log1p(cat_cardinality_sum)),
        "log_cat_cardinality_max": float(np.log1p(cat_cardinality_max)),
        "missing_rate": float(missing_rate),
    }

    dataset_name = str(payload.get("dataset_name", "") or "")
    stage_name = str(payload.get("stage_name", "") or "")
    method_name = str(payload.get("method_name", "") or "")
    if dataset_name:
        features[f"dataset={dataset_name}"] = 1.0
    if stage_name:
        features[f"stage={stage_name}"] = 1.0
    if method_name:
        features[f"method={method_name}"] = 1.0
    return features


def machine_cost_calibration_multiplier(cost_param_payload=None, base_dir=None):
    payload = dict(cost_param_payload or {})
    mode = str(payload.get("mode", "e")).lower()
    calibrator = load_machine_cost_calibrator(base_dir=base_dir, cost_param=mode)
    if not calibrator:
        return 1.0

    feature_names = list(calibrator.get("feature_names", []))
    coefficients = np.asarray(calibrator.get("coefficients", []), dtype=float)
    if not feature_names or len(feature_names) != len(coefficients):
        return 1.0

    raw_features = build_machine_cost_calibration_features(payload)
    values = np.asarray([float(raw_features.get(name, 0.0)) for name in feature_names], dtype=float)
    log_multiplier = float(np.dot(values, coefficients))
    multiplier = float(np.exp(log_multiplier))
    clip_min = float(calibrator.get("clip_min", 0.2))
    clip_max = float(calibrator.get("clip_max", 5.0))
    return float(np.clip(multiplier, clip_min, clip_max))


def get_effective_time_coeff(
    time_coeff,
    cost_param_payload=None,
    base_dir=None,
    apply_calibration=True,
    coeff_mode="scaled",
):
    coeff = list(time_coeff or [0.0, 0.0])
    if len(coeff) < 2:
        coeff = coeff + [0.0] * (2 - len(coeff))
    coeff_mode = str(coeff_mode or "scaled").strip().lower()
    if coeff_mode == "profiled":
        return [float(coeff[0]), float(coeff[1])]
    if coeff_mode != "scaled":
        raise ValueError("coeff_mode must be 'scaled' or 'profiled'.")
    mult = compute_cost_param_multiplier(cost_param_payload)
    if apply_calibration:
        mult *= machine_cost_calibration_multiplier(cost_param_payload, base_dir=base_dir)
    return [float(coeff[0]) * mult, float(coeff[1]) * mult]


def stage_machine_time(stage, e_val):
    coeff = stage.get("effective_time_coeff") or get_effective_time_coeff(
        stage.get("time_coeff", [0.0, 0.0]),
        stage.get("cost_param_payload", {}),
    )
    return Cost.machine_time(coeff, e_val)


def enrich_config_by_cost_param(config, stages=None, cost_param="e", dataset_name=None):
    cfg = dict(config or {})
    cost_param = str(cost_param or cfg.get("cost_param", "e")).lower()
    cfg["cost_param"] = cost_param
    dataset_name = resolve_dataset_name(dataset_name or cfg.get("dataset"))
    if dataset_name:
        cfg["dataset"] = dataset_name

    if cost_param in ("e_dataset", "e_dataset_method"):
        ds_params = get_dataset_cost_params(dataset_name)
        for key, value in ds_params.items():
            cfg[key] = value

    if cost_param == "e_dataset_method" and stages:
        for stage in stages:
            stage_name = stage.get("name")
            method_name = stage.get("method")
            payload = build_cost_param_payload(stage_name, method_name, cost_param="e_dataset_method", dataset_name=None)
            for key, value in payload.items():
                if key in ("mode", "method_name", "dataset_rows", "dataset_cols", "e"):
                    continue
                if key in ("dataset_name", "stage_name"):
                    continue
                cfg[f"{stage_name}_{key}"] = value
    return cfg


def build_time_coeff_config(
    methods,
    total_time,
    max_samples,
    model,
    step=0.1,
    base_dir=None,
    verbose=True,
    dataset=None,
    cost_param="e",
    cost_coeff_mode="scaled",
    runtime_model_version=None,
    runtime_cost_mode="mean",
):
    """
    methods: dict mapping stage name -> method string
    returns config for optimizer with stages including time_coeff
    """
    dataset = resolve_dataset_name(dataset) if dataset else None
    dataset_cost_params = {}
    if str(cost_param or "e").lower() in ("e_dataset", "e_dataset_method"):
        dataset_cost_params = get_dataset_cost_params(dataset)
    stages = []
    for stage_name, method in methods.items():
        if runtime_model_version:
            time_coeff, runtime_profile = get_versioned_runtime_coeff(
                dataset=dataset,
                stage_name=stage_name,
                method=method,
                cost_param=cost_param,
                runtime_mode=runtime_cost_mode,
                base_dir=base_dir,
                version=runtime_model_version,
            )
            time_file = (
                f"artifacts/cost_models/{runtime_model_version}/runtime_cost_models.pkl"
            )
            if stage_name in ("data_conversion", "dataset_division"):
                e_range = (0.0, 0.0)
            else:
                e_range = (0.1, 1.0)
            if verbose:
                print(
                    f"[time_coeff] stage={stage_name} method={method} "
                    f"version={runtime_model_version} mode={runtime_cost_mode} coeff={time_coeff}"
                )
        else:
            strict_cost_param = str(cost_param or "e").lower() != "e"
            time_file = _find_time_file(
                method,
                step,
                base_dir=base_dir,
                dataset=dataset,
                cost_param=cost_param,
                strict_cost_param=strict_cost_param,
            )
            if not time_file:
                raise FileNotFoundError(f"time file not found for method '{method}' (step={step}, dataset={dataset})")
            time_list = pickle.load(open(time_file, "rb"))
            if verbose:
                print(f"[time_coeff] stage={stage_name} method={method} file={time_file}")
                print(f"[time_coeff] samples={len(time_list)} head={time_list[:5] if len(time_list) > 5 else time_list}")
            if time_file.endswith("_full.pkl") or "_full_" in os.path.basename(time_file):
                avg_time = float(np.mean(time_list)) if len(time_list) > 0 else 0.0
                time_coeff = [0.0, max(0.0, avg_time)]
                e_range = (0.0, 0.0)
            else:
                time_coeff = _fit_time_coeff(time_list, step)
                e_range = (0.1, 1.0)
        if verbose:
            print(f"[time_coeff] coeff={time_coeff}")
        cost_param_payload = build_cost_param_payload(stage_name, method, cost_param=cost_param, dataset_name=dataset)
        if runtime_model_version:
            # Versioned profiles are already fitted for a concrete dataset and method.
            effective_time_coeff = list(time_coeff)
        else:
            effective_time_coeff = get_effective_time_coeff(
                time_coeff,
                cost_param_payload,
                base_dir=base_dir,
                coeff_mode=cost_coeff_mode,
            )
        stages.append({
            'name': stage_name,
            'method': method,
            'e_range': e_range,
            'time_coeff': time_coeff,
            'cost_param_payload': cost_param_payload,
            'effective_time_coeff': effective_time_coeff,
        })
    return {
        'stages': stages,
        'total_time': total_time,
        'max_samples': max_samples,
        'model': model,
        'dataset': dataset,
        'cost_param': str(cost_param or "e").lower(),
        'cost_coeff_mode': str(cost_coeff_mode or "scaled").lower(),
        'runtime_model_version': runtime_model_version,
        'runtime_cost_mode': str(runtime_cost_mode or "mean").lower(),
        'dataset_cost_params': dataset_cost_params,
    }


if __name__ == "__main__":
    # 函数验证测试
    print("机器时间测试:", Cost.machine_time(0.5, 20))  # 应输出11.25
    print("机器成本测试:", Cost.machine_cost(0.3, 100))  # 约116.18
    print("人工时间测试:", Cost.human_time('feature_selection', 4.0, 10.0,100,200,0.5,0.7,4))  # 应输出36.0 task_type, d, d_max, n, n_max, mu ,e,number
    print("人工成本测试:", Cost.human_cost('data_labeling', 8.0, 50.0,500,1000,0.2,0.9,30))  # 应输出180.0
