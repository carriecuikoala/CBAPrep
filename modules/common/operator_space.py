from copy import deepcopy


SOTA_OPERATOR_VERSION = "sota_v1"

LEGACY_METHOD_OPTIONS = {
    "feature_encoding": ["ordinal", "onehot", "count", "target"],
    "imputation": ["DROP", "MEAN", "MEDIAN", "MF", "MICE", "KNN"],
    "data_normalization": [
        "z_score",
        "robust",
        "min_max",
        "max_abs",
        "normalizer",
        "sigmoid",
    ],
    "outlier_detection": ["ZSB", "IQR", "LOF"],
    "feature_extraction": ["pca", "truncated_svd", "cfs"],
    "feature_selection": ["LC", "VAR", "Tree", "SVC"],
}

SOTA_METHOD_ADDITIONS = {
    "duplicate_detection": ["ED", "AD"],
    "imputation": ["EM", "DT", "DUMMY"],
    "data_normalization": [
        "quantile_uniform",
        "quantile_normal",
        "power",
        "kbins_uniform",
        "kbins_quantile",
    ],
    "outlier_detection": ["MAD"],
    "feature_extraction": [
        "polynomial",
        "interaction",
        "incremental_pca",
        "kernel_pca",
        "random_trees_embedding",
        "pca_arpack",
        "pca_randomized",
    ],
    "feature_selection": ["MR", "WR"],
}


def normalize_operator_space(operator_space):
    value = str(operator_space or "legacy").strip().lower()
    if value in {"legacy", "original", "base"}:
        return "legacy"
    if value in {"sota", "sota_v1", "extended"}:
        return "sota"
    raise ValueError("operator_space must be 'legacy' or 'sota'.")


def get_method_options(operator_space="legacy", include_optional=False):
    space = normalize_operator_space(operator_space)
    options = deepcopy(LEGACY_METHOD_OPTIONS)
    if space == "legacy":
        if include_optional:
            return {stage: [None] + methods for stage, methods in options.items()}
        return options

    ordered = {
        "duplicate_detection": SOTA_METHOD_ADDITIONS["duplicate_detection"][:],
        "feature_encoding": options["feature_encoding"],
        "imputation": options["imputation"] + SOTA_METHOD_ADDITIONS["imputation"],
        "data_normalization": (
            options["data_normalization"] + SOTA_METHOD_ADDITIONS["data_normalization"]
        ),
        "outlier_detection": (
            options["outlier_detection"] + SOTA_METHOD_ADDITIONS["outlier_detection"]
        ),
        "feature_extraction": (
            options["feature_extraction"] + SOTA_METHOD_ADDITIONS["feature_extraction"]
        ),
        "feature_selection": (
            options["feature_selection"] + SOTA_METHOD_ADDITIONS["feature_selection"]
        ),
    }
    if include_optional:
        return {stage: [None] + methods for stage, methods in ordered.items()}
    return ordered


def get_method_order(operator_space="legacy"):
    options = get_method_options(operator_space)
    order = []
    for methods in options.values():
        order.extend(methods)
    return order


def get_artifact_suffix(cost_param="e", operator_space="legacy", artifact_version=None):
    cost_param = str(cost_param or "e").strip().lower()
    if normalize_operator_space(operator_space) == "legacy":
        return cost_param
    version = str(artifact_version or SOTA_OPERATOR_VERSION).strip().lower()
    if not version.replace("_", "").isalnum():
        raise ValueError("artifact_version may contain only letters, numbers, and underscores.")
    return f"{cost_param}_{version}"
