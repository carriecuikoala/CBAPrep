import argparse
import contextlib
import itertools
import io
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.benefit_predictor.acc_predictor import acc, acc_batch
from modules.common.dataset_utils import find_project_root, load_dataset_label, resolve_dataset_name, resolve_dataset_paths
from modules.common.operator_space import get_method_options, normalize_operator_space
from modules.cost_calculator.cost_calculator import (
    build_cost_param_payload,
    build_time_coeff_config,
    enrich_config_by_cost_param,
    get_effective_time_coeff,
    stage_machine_time,
)
from modules.optimizer.searcher import search_best_pipeline, _run_actual_pipeline
from modules.methods.preprocessing.data_conversion.data_converter import DataConverter
from modules.methods.preprocessing.dataset_division.dataset_divider import DatasetDivider
from modules.ml_modules.Classification.classifier import Classifier


STAGE_OPTIONS = {
    "feature_encoding": ["ordinal", "onehot", "count", "target"],
    "imputation": ["DROP", "MEAN", "MEDIAN", "MF", "MICE", "KNN"],
    "data_normalization": ["z_score", "robust", "min_max", "max_abs", "normalizer", "sigmoid"],
    "outlier_detection": ["ZSB", "IQR", "LOF"],
    "feature_extraction": ["pca", "truncated_svd", "cfs"],
    "feature_selection": ["LC", "VAR", "Tree", "SVC"],
}

STAGE_TO_KEYS = {
    "feature_encoding": ("method_feature_encoding", "e_feature_encoding"),
    "imputation": ("method_imputation", "e_imputation"),
    "data_normalization": ("method_data_normalization", "e_data_normalization"),
    "outlier_detection": ("method_outlier_detection", "e_outlier_detection"),
    "feature_extraction": ("method_feature_extraction", "e_feature_extraction"),
    "feature_selection": ("method_feature_selection", "e_feature_selection"),
}

BASELINES = [
    "CBAprep",
    "Random-Budget Search",
    "Greedy-Knapsack",
    "Uniform-Budget Allocation",
]

BUDGET_AWARE_BASELINES = [
    "Random-Budget Search",
    "Greedy-Knapsack",
    "Uniform-Budget Allocation",
]

METHOD_ALIASES = {
    "cbaprep": "CBAprep",
    "cba": "CBAprep",
    "random": "Random-Budget Search",
    "random-budget": "Random-Budget Search",
    "random-budget-search": "Random-Budget Search",
    "random budget search": "Random-Budget Search",
    "greedy": "Greedy-Knapsack",
    "greedy-knapsack": "Greedy-Knapsack",
    "greedy knapsack": "Greedy-Knapsack",
    "uniform": "Uniform-Budget Allocation",
    "uniform-budget": "Uniform-Budget Allocation",
    "uniform-budget-allocation": "Uniform-Budget Allocation",
    "uniform budget allocation": "Uniform-Budget Allocation",
}


def parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def normalize_method_name(name):
    raw = str(name).strip()
    key = raw.lower().replace("_", "-")
    key = " ".join(key.split())
    if key in METHOD_ALIASES:
        return METHOD_ALIASES[key]
    key_dash = key.replace(" ", "-")
    if key_dash in METHOD_ALIASES:
        return METHOD_ALIASES[key_dash]
    for method_name in BASELINES:
        if raw.lower() == method_name.lower():
            return method_name
    raise ValueError(f"Unknown method '{name}'. Valid methods: {', '.join(BASELINES)}")


def get_active_methods(args):
    methods_text = getattr(args, "methods", None)
    if methods_text:
        methods = [normalize_method_name(x) for x in str(methods_text).split(",") if x.strip()]
    elif getattr(args, "budget_baselines_only", False):
        methods = BUDGET_AWARE_BASELINES[:]
    else:
        methods = BASELINES[:]

    deduped = []
    for method_name in methods:
        if method_name not in deduped:
            deduped.append(method_name)
    if not deduped:
        raise ValueError("No experiment methods selected.")
    return deduped


def get_output_dir(project_root):
    out_dir = os.path.join(project_root, "artifacts", "experiments", "budget", "baselines")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def build_time_cache(project_root, dataset_name, model_name, cost_param, operator_space="legacy"):
    stage_options = get_method_options(normalize_operator_space(operator_space))
    methods = sorted({m for values in stage_options.values() for m in values})
    methods.extend(["data_conversion", "dataset_division"])
    raw_cache = {}
    for method in methods:
        cfg = build_time_coeff_config(
            methods={"tmp": method},
            total_time=1.0,
            max_samples=1,
            model=model_name,
            verbose=False,
            base_dir=project_root,
            dataset=dataset_name,
            cost_param=cost_param,
        )
        raw_cache[method] = cfg["stages"][0]["time_coeff"]

    cache = {}
    for stage_name, methods_for_stage in stage_options.items():
        cache[stage_name] = {}
        for method in methods_for_stage:
            payload = build_cost_param_payload(
                stage_name,
                method,
                cost_param=cost_param,
                dataset_name=dataset_name,
            )
            coeff = raw_cache[method]
            cache[stage_name][method] = {
                "name": stage_name,
                "method": method,
                "time_coeff": coeff,
                "cost_param_payload": payload,
                "effective_time_coeff": get_effective_time_coeff(coeff, payload),
            }

    overhead = []
    for stage_name, method in [
        ("data_conversion", "data_conversion"),
        ("dataset_division", "dataset_division"),
    ]:
        payload = build_cost_param_payload(
            stage_name,
            method,
            cost_param=cost_param,
            dataset_name=dataset_name,
        )
        coeff = raw_cache[method]
        overhead.append({
            "name": stage_name,
            "method": method,
            "time_coeff": coeff,
            "cost_param_payload": payload,
            "effective_time_coeff": get_effective_time_coeff(coeff, payload),
        })
    return cache, overhead


def stage_time(time_cache, stage_name, method, e_value):
    return float(stage_machine_time(time_cache[stage_name][method], float(e_value)))


def overhead_time(overhead_stages):
    return float(sum(stage_machine_time(stage, 0.0) for stage in overhead_stages))


def total_pred_time(time_cache, overhead_stages, methods, e_values):
    total = overhead_time(overhead_stages)
    for stage_name, method in methods.items():
        total += stage_time(time_cache, stage_name, method, e_values.get(stage_name, 0.0))
    return float(total)


def build_pred_config(
    methods,
    e_values,
    dataset_name,
    cost_param,
    time_cache=None,
    benefit_model="specific",
    model_name=None,
    operator_space="legacy",
):
    cfg = {
        "dataset": dataset_name,
        "cost_param": cost_param,
        "benefit_model": benefit_model,
        "operator_space": normalize_operator_space(operator_space),
    }
    if model_name:
        cfg["model"] = model_name
    stages = []
    for stage_name, method in methods.items():
        method_key, e_key = STAGE_TO_KEYS[stage_name]
        cfg[method_key] = method
        cfg[e_key] = float(e_values.get(stage_name, 0.0))
        if time_cache is not None:
            stages.append(time_cache[stage_name][method])
    return enrich_config_by_cost_param(
        cfg,
        stages=stages,
        cost_param=cost_param,
        dataset_name=dataset_name,
    )


def predicted_accuracy(
    methods,
    e_values,
    dataset_name,
    model_name,
    cost_param,
    time_cache=None,
    benefit_model="specific",
    operator_space="legacy",
):
    cfg = build_pred_config(
        methods,
        e_values,
        dataset_name,
        cost_param,
        time_cache=time_cache,
        benefit_model=benefit_model,
        model_name=model_name,
        operator_space=operator_space,
    )
    return float(acc(cfg, model_name=model_name, verbose=False, benefit_model=benefit_model))


def predicted_accuracy_batch(
    plans,
    dataset_name,
    model_name,
    cost_param,
    time_cache=None,
    benefit_model="specific",
    operator_space="legacy",
):
    configs = [
        build_pred_config(
            methods,
            e_values,
            dataset_name,
            cost_param,
            time_cache=time_cache,
            benefit_model=benefit_model,
            model_name=model_name,
            operator_space=operator_space,
        )
        for methods, e_values in plans
    ]
    return acc_batch(configs, model_name=model_name, benefit_model=benefit_model)


def all_method_combos(max_combos, rng, random_sample=True):
    stage_names = list(STAGE_OPTIONS.keys())
    combos = list(itertools.product(*(STAGE_OPTIONS[s] for s in stage_names)))
    if random_sample and max_combos and max_combos < len(combos):
        idx = rng.choice(len(combos), size=max_combos, replace=False)
        combos = [combos[i] for i in sorted(idx)]
    elif max_combos:
        combos = combos[:max_combos]
    return [dict(zip(stage_names, combo)) for combo in combos]


def cheapest_methods(time_cache, e_min):
    methods = {}
    for stage_name, options in STAGE_OPTIONS.items():
        methods[stage_name] = min(
            options,
            key=lambda method: stage_time(time_cache, stage_name, method, e_min),
        )
    return methods


def cheapest_config(time_cache, e_min):
    methods = cheapest_methods(time_cache, e_min)
    e_values = {stage_name: e_min for stage_name in STAGE_OPTIONS}
    return methods, e_values


def random_budget_search(time_cache, overhead_stages, dataset_name, model_name, budget, samples, e_min, cost_param, rng, benefit_model="specific"):
    best = None
    sampled_candidates = 0
    budget_feasible_candidates = 0
    valid_evaluated_candidates = 0
    failed_candidates = 0
    for i in range(samples):
        sampled_candidates += 1
        methods = {
            stage_name: rng.choice(options)
            for stage_name, options in STAGE_OPTIONS.items()
        }
        e_values = {
            stage_name: float(rng.uniform(e_min, 1.0))
            for stage_name in STAGE_OPTIONS
        }
        pred_time = total_pred_time(time_cache, overhead_stages, methods, e_values)
        if pred_time > budget:
            continue
        budget_feasible_candidates += 1
        try:
            val_acc, val_f1, val_time, _, _ = _run_actual_pipeline(
                methods,
                e_values,
                model_name=model_name,
                dataset_name=dataset_name,
                target_name=None,
                eval_split="val",
            )
        except Exception:
            failed_candidates += 1
            continue
        if val_acc is None or not np.isfinite(float(val_acc)):
            failed_candidates += 1
            continue
        valid_evaluated_candidates += 1
        pred_acc = predicted_accuracy(methods, e_values, dataset_name, model_name, cost_param, time_cache, benefit_model=benefit_model)
        if valid_evaluated_candidates == 1 or valid_evaluated_candidates % 10 == 0 or i == samples - 1:
            print(
                f"  random validation candidates: "
                f"{valid_evaluated_candidates}/{budget_feasible_candidates} "
                f"(sampled={sampled_candidates})",
                flush=True,
            )
        if (
            best is None
            or float(val_acc) > best["selection_acc"]
            or (
                float(val_acc) == best["selection_acc"]
                and float(val_time) < best["selection_time"]
            )
        ):
            best = {
                "methods": methods,
                "e_values": e_values,
                "pred_time": pred_time,
                "pred_acc": pred_acc,
                "selection_acc": float(val_acc),
                "selection_f1": float(val_f1) if val_f1 is not None else np.nan,
                "selection_time": float(val_time) if val_time is not None else np.nan,
                "sampled_candidates": sampled_candidates,
                "budget_feasible_candidates": budget_feasible_candidates,
                "valid_evaluated_candidates": valid_evaluated_candidates,
                "failed_candidates": failed_candidates,
                "feasible": True,
            }
    if best is not None:
        best["sampled_candidates"] = sampled_candidates
        best["budget_feasible_candidates"] = budget_feasible_candidates
        best["valid_evaluated_candidates"] = valid_evaluated_candidates
        best["failed_candidates"] = failed_candidates
        return best

    methods, e_values = cheapest_config(time_cache, e_min)
    pred_time = total_pred_time(time_cache, overhead_stages, methods, e_values)
    return {
        "methods": methods,
        "e_values": e_values,
        "pred_time": pred_time,
        "pred_acc": predicted_accuracy(methods, e_values, dataset_name, model_name, cost_param, time_cache, benefit_model=benefit_model),
        "selection_acc": np.nan,
        "selection_f1": np.nan,
        "selection_time": np.nan,
        "sampled_candidates": sampled_candidates,
        "budget_feasible_candidates": budget_feasible_candidates,
        "valid_evaluated_candidates": valid_evaluated_candidates,
        "failed_candidates": failed_candidates,
        "feasible": False,
    }


def greedy_knapsack(time_cache, overhead_stages, dataset_name, model_name, budget, e_min, e_grid, cost_param, benefit_model="specific"):
    base_methods, base_e_values = cheapest_config(time_cache, e_min)
    current_methods = dict(base_methods)
    current_e_values = dict(base_e_values)
    base_time = total_pred_time(time_cache, overhead_stages, base_methods, base_e_values)

    candidate_plans = [(base_methods, base_e_values)]
    candidate_meta = [None]
    for stage_name, options in STAGE_OPTIONS.items():
        for method in options:
            for e_value in e_grid:
                candidate_methods = dict(base_methods)
                candidate_e_values = dict(base_e_values)
                candidate_methods[stage_name] = method
                candidate_e_values[stage_name] = float(e_value)
                pred_time = total_pred_time(time_cache, overhead_stages, candidate_methods, candidate_e_values)
                candidate_plans.append((candidate_methods, candidate_e_values))
                candidate_meta.append({
                    "stage": stage_name,
                    "method": method,
                    "e": float(e_value),
                    "pred_time": pred_time,
                })

    predictions = predicted_accuracy_batch(candidate_plans, dataset_name, model_name, cost_param, time_cache, benefit_model=benefit_model)
    base_acc = float(predictions[0])
    items = []
    for meta, pred_acc in zip(candidate_meta[1:], predictions[1:]):
        cost_delta = max(0.0, float(meta["pred_time"]) - base_time)
        gain_delta = max(0.0, float(pred_acc) - base_acc)
        ratio = gain_delta / max(cost_delta, 1e-9)
        item = dict(meta)
        item.update({
            "cost_delta": cost_delta,
            "gain_delta": gain_delta,
            "ratio": ratio,
        })
        items.append(item)

    items.sort(key=lambda x: (x["ratio"], x["gain_delta"]), reverse=True)
    current_time = base_time
    selected_stages = set()
    for item in items:
        stage_name = item["stage"]
        if stage_name in selected_stages:
            continue
        candidate_methods = dict(current_methods)
        candidate_e_values = dict(current_e_values)
        candidate_methods[stage_name] = item["method"]
        candidate_e_values[stage_name] = item["e"]
        candidate_time = total_pred_time(time_cache, overhead_stages, candidate_methods, candidate_e_values)
        if candidate_time <= budget:
            current_methods = candidate_methods
            current_e_values = candidate_e_values
            current_time = candidate_time
            selected_stages.add(stage_name)

    return {
        "methods": current_methods,
        "e_values": current_e_values,
        "pred_time": current_time,
        "pred_acc": predicted_accuracy(current_methods, current_e_values, dataset_name, model_name, cost_param, time_cache, benefit_model=benefit_model),
        "feasible": current_time <= budget,
    }


def uniform_allocate(time_cache, overhead_stages, methods, budget, e_min):
    e_values = {stage_name: e_min for stage_name in STAGE_OPTIONS}
    base_time = total_pred_time(time_cache, overhead_stages, methods, e_values)
    if base_time >= budget:
        return e_values, base_time

    remaining = float(budget - base_time)
    active = set(STAGE_OPTIONS.keys())
    while active and remaining > 1e-12:
        share = remaining / len(active)
        consumed = 0.0
        saturated = []
        for stage_name in list(active):
            method = methods[stage_name]
            cur_e = e_values[stage_name]
            cur_cost = stage_time(time_cache, stage_name, method, cur_e)
            max_cost = stage_time(time_cache, stage_name, method, 1.0)
            increment = max(0.0, max_cost - cur_cost)
            if increment <= 1e-12:
                e_values[stage_name] = 1.0
                saturated.append(stage_name)
                continue
            use = min(share, increment)
            slope = float(time_cache[stage_name][method]["effective_time_coeff"][0])
            if slope > 1e-12:
                e_values[stage_name] = min(1.0, cur_e + use / slope)
            else:
                e_values[stage_name] = 1.0
            consumed += use
            if use >= increment - 1e-12:
                saturated.append(stage_name)
        for stage_name in saturated:
            active.discard(stage_name)
        if consumed <= 1e-12:
            break
        remaining -= consumed
    pred_time = total_pred_time(time_cache, overhead_stages, methods, e_values)
    return e_values, pred_time


def uniform_budget_allocation(
    time_cache,
    overhead_stages,
    dataset_name,
    model_name,
    budget,
    max_combos,
    e_min,
    cost_param,
    rng,
    random_sample=True,
    benefit_model="specific",
):
    feasible_plans = []
    feasible_times = []
    for methods in all_method_combos(max_combos, rng, random_sample=random_sample):
        e_values, pred_time = uniform_allocate(time_cache, overhead_stages, methods, budget, e_min)
        if pred_time > budget:
            continue
        feasible_plans.append((methods, e_values))
        feasible_times.append(float(pred_time))
    if feasible_plans:
        predictions = predicted_accuracy_batch(feasible_plans, dataset_name, model_name, cost_param, time_cache, benefit_model=benefit_model)
        best_idx = int(np.nanargmax(predictions))
        methods, e_values = feasible_plans[best_idx]
        return {
            "methods": methods,
            "e_values": e_values,
            "pred_time": feasible_times[best_idx],
            "pred_acc": float(predictions[best_idx]),
            "feasible": True,
        }

    methods, e_values = cheapest_config(time_cache, e_min)
    return {
        "methods": methods,
        "e_values": e_values,
        "pred_time": total_pred_time(time_cache, overhead_stages, methods, e_values),
        "pred_acc": predicted_accuracy(methods, e_values, dataset_name, model_name, cost_param, time_cache, benefit_model=benefit_model),
        "feasible": False,
    }


def cbaprep_search(
    dataset_name,
    model_name,
    budget,
    max_samples,
    max_combos,
    cost_param,
    optimizer,
    search_mode,
    benefit_model="specific",
    random_state=42,
    restarts=1,
    validation_rerank=True,
    budget_guard=0.95,
):
    """Search several independent surrogate optima and rerank on validation data.

    The held-out test set is deliberately excluded here.  It is evaluated once by
    ``validate_plan`` after a candidate has been selected.  Multiple starts reduce
    sensitivity to the random subset of discrete operator combinations, while the
    validation rerank limits exploitation of local GBDT prediction errors.
    """
    candidates = []
    restarts = max(1, int(restarts))
    for restart in range(restarts):
        seed = int(random_state) + 101 * restart
        result = search_best_pipeline(
            total_time=float(budget),
            max_samples=int(max_samples),
            model_name=model_name,
            max_combos=int(max_combos),
            validate=False,
            random_sample=True,
            dataset_name=dataset_name,
            target_name=None,
            search_mode=search_mode,
            allow_infeasible=True,
            optimizer_name=optimizer,
            cost_param=cost_param,
            benefit_model=benefit_model,
            random_state=seed,
        )
        methods = result.get("methods")
        e_values = result.get("e_values")
        pred_acc = result.get("pred_acc", np.nan)
        pred_time = result.get("pred_time", np.nan)
        if not methods or not e_values or not np.isfinite(pred_acc):
            continue

        candidate = {
            "methods": methods,
            "e_values": e_values,
            "pred_time": float(pred_time),
            "pred_acc": float(pred_acc),
            "restart": restart,
            "seed": seed,
            "feasible": bool(np.isfinite(pred_time) and float(pred_time) <= float(budget)),
        }
        if validation_rerank:
            try:
                val_acc, val_f1, val_time, _, _ = _run_actual_pipeline(
                    methods,
                    e_values,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    target_name=None,
                    eval_split="val",
                )
                candidate.update({
                    "selection_acc": float(val_acc) if val_acc is not None else np.nan,
                    "selection_f1": float(val_f1) if val_f1 is not None else np.nan,
                    "selection_time": float(val_time) if val_time is not None else np.nan,
                })
            except Exception as exc:
                candidate.update({
                    "selection_acc": np.nan,
                    "selection_f1": np.nan,
                    "selection_time": np.nan,
                    "selection_error": str(exc),
                })
        candidates.append(candidate)

    if not candidates:
        return {
            "methods": None,
            "e_values": None,
            "pred_time": np.inf,
            "pred_acc": -np.inf,
            "feasible": False,
            "candidate_restarts": restarts,
            "valid_evaluated_candidates": 0,
        }

    if validation_rerank:
        guarded_budget = float(budget) * float(budget_guard)
        valid = [
            candidate
            for candidate in candidates
            if np.isfinite(candidate.get("selection_acc", np.nan))
            and np.isfinite(candidate.get("selection_time", np.nan))
            and candidate["selection_time"] <= guarded_budget
        ]
        if not valid:
            # Preserve a usable result when timing noise makes every validation
            # run cross the guard, but never prefer a plan that exceeds B itself.
            valid = [
                candidate
                for candidate in candidates
                if np.isfinite(candidate.get("selection_acc", np.nan))
                and np.isfinite(candidate.get("selection_time", np.nan))
                and candidate["selection_time"] <= float(budget)
            ]
        if valid:
            best = max(
                valid,
                key=lambda candidate: (
                    candidate["selection_acc"],
                    -candidate["selection_time"],
                ),
            )
        else:
            best = min(candidates, key=lambda candidate: candidate.get("pred_time", np.inf))
            best["feasible"] = False
    else:
        best = max(candidates, key=lambda candidate: candidate.get("pred_acc", -np.inf))

    best = dict(best)
    best["candidate_restarts"] = restarts
    best["valid_evaluated_candidates"] = sum(
        np.isfinite(candidate.get("selection_acc", np.nan)) for candidate in candidates
    )
    best["budget_feasible_candidates"] = sum(
        np.isfinite(candidate.get("selection_time", np.nan))
        and candidate.get("selection_time", np.inf) <= float(budget)
        for candidate in candidates
    )
    return best


def validate_plan(plan, dataset_name, model_name, repeats=3):
    if not plan.get("methods") or not plan.get("e_values"):
        return {
            "actual_acc": np.nan,
            "actual_f1": np.nan,
            "actual_time": np.nan,
            "actual_acc_std": np.nan,
            "actual_time_std": np.nan,
            "actual_time_samples": [],
            "step_times": {},
            "data_stats": {},
        }
    measurements = []
    repeats = max(1, int(repeats))
    for _ in range(repeats):
        measurements.append(
            _run_actual_pipeline(
                plan["methods"],
                plan["e_values"],
                model_name=model_name,
                dataset_name=dataset_name,
                target_name=None,
            )
        )
    acc_values = np.asarray([item[0] for item in measurements], dtype=float)
    f1_values = np.asarray([item[1] for item in measurements], dtype=float)
    time_values = np.asarray([item[2] for item in measurements], dtype=float)
    median_index = int(np.argsort(time_values)[len(time_values) // 2])
    step_times = measurements[median_index][3]
    data_stats = measurements[median_index][4]
    return {
        "actual_acc": float(np.nanmedian(acc_values)),
        "actual_f1": float(np.nanmedian(f1_values)),
        "actual_time": float(np.nanmedian(time_values)),
        "actual_acc_std": float(np.nanstd(acc_values)),
        "actual_time_std": float(np.nanstd(time_values)),
        "actual_time_samples": [float(value) for value in time_values],
        "step_times": step_times,
        "data_stats": data_stats,
    }


def no_optional_baseline(project_root, dataset_name, model_name):
    _, data_path, _ = resolve_dataset_paths(project_root, dataset_name)
    _, target_col = load_dataset_label(project_root, dataset_name)
    with contextlib.redirect_stdout(io.StringIO()):
        raw = pd.read_csv(data_path)
        t0 = time.perf_counter()
        data, _ = DataConverter(raw).transform()
        train, _, test, _ = DatasetDivider(data, test_rate=0.3, val_rate=0.3).transform()

        X_train = train.drop(columns=[target_col]).select_dtypes(include=[np.number])
        y_train = train[target_col]
        X_test = test.drop(columns=[target_col]).select_dtypes(include=[np.number])
        y_test = test[target_col]

        cols = X_train.columns.tolist()
        X_test = X_test.reindex(columns=cols, fill_value=0.0)
        med = X_train.median(numeric_only=True).fillna(0.0)
        X_train = X_train.fillna(med).fillna(0.0)
        X_test = X_test.fillna(med).fillna(0.0)

        dataset = {
            "train": X_train.join(y_train),
            "test": X_test.join(y_test),
            "target": y_train,
            "target_test": y_test,
        }
        cl = Classifier(dataset, target=target_col, strategy=model_name, k_folds=10, verbose=False)
        metrics = cl.transform(return_metrics=True)
        elapsed = time.perf_counter() - t0
    return (
        float(metrics.get("accuracy", np.nan)),
        float(elapsed),
        float(metrics.get("f1", np.nan)) if metrics.get("f1") is not None else np.nan,
    )


def flatten_methods(methods):
    return ";".join(f"{k}={v}" for k, v in sorted((methods or {}).items()))


def flatten_e_values(e_values):
    return ";".join(f"{k}={float(v):.4f}" for k, v in sorted((e_values or {}).items()))


def add_record(records, dataset_name, model_name, budget, method_name, plan, validation, base_acc, benefit_model="specific"):
    actual_acc = validation["actual_acc"]
    actual_time = validation["actual_time"]
    gain = actual_acc - base_acc if np.isfinite(actual_acc) and np.isfinite(base_acc) else np.nan
    gain_per_cost = gain / actual_time if np.isfinite(gain) and np.isfinite(actual_time) and actual_time > 0 else np.nan
    budget_utilization = actual_time / budget if np.isfinite(actual_time) and budget > 0 else np.nan
    records.append({
        "dataset": dataset_name,
        "model": model_name,
        "benefit_model": benefit_model,
        "method": method_name,
        "budget": float(budget),
        "actual_acc": actual_acc,
        "actual_f1": validation["actual_f1"],
        "actual_time": actual_time,
        "actual_acc_std": validation.get("actual_acc_std", np.nan),
        "actual_time_std": validation.get("actual_time_std", np.nan),
        "actual_time_samples": ";".join(
            f"{value:.6f}" for value in validation.get("actual_time_samples", [])
        ),
        "budget_utilization": budget_utilization,
        "gain": gain,
        "gain_per_cost": gain_per_cost,
        "pred_acc": plan.get("pred_acc", np.nan),
        "pred_time": plan.get("pred_time", np.nan),
        "pred_feasible": bool(plan.get("feasible", False)),
        "selection_acc": plan.get("selection_acc", np.nan),
        "selection_f1": plan.get("selection_f1", np.nan),
        "selection_time": plan.get("selection_time", np.nan),
        "sampled_candidates": plan.get("sampled_candidates", np.nan),
        "budget_feasible_candidates": plan.get("budget_feasible_candidates", np.nan),
        "valid_evaluated_candidates": plan.get("valid_evaluated_candidates", np.nan),
        "failed_candidates": plan.get("failed_candidates", np.nan),
        "candidate_restarts": plan.get("candidate_restarts", np.nan),
        "selected_restart": plan.get("restart", np.nan),
        "selected_seed": plan.get("seed", np.nan),
        "methods": flatten_methods(plan.get("methods", {})),
        "e_values": flatten_e_values(plan.get("e_values", {})),
    })


def plot_metric(df, metric, out_dir, dataset_name, methods=None):
    methods = methods or BASELINES
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for method_name in methods:
        part = df[df["method"] == method_name].sort_values("budget")
        if part.empty:
            continue
        ax.plot(part["budget"], part[metric], marker="o", linewidth=1.8, label=method_name)
    ax.set_xlabel("Budget")
    ax.set_ylabel(metric)
    ax.set_title(f"{dataset_name}: {metric} under budget constraints")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{dataset_name}_{metric}.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def run_experiment(args):
    project_root = find_project_root(os.path.dirname(__file__))
    if project_root is None:
        raise FileNotFoundError("Cannot locate project root.")
    out_dir = get_output_dir(project_root)
    rng = np.random.default_rng(int(args.random_state))

    datasets = [resolve_dataset_name(x.strip()) for x in str(args.datasets).split(",") if x.strip()]
    budgets = parse_float_list(args.budgets)
    e_grid = parse_float_list(args.e_grid)
    model_name = args.model.upper()
    cost_param = str(args.cost_param).lower()
    benefit_model = str(getattr(args, "benefit_model", "specific")).lower()
    active_methods = get_active_methods(args)
    records = []
    print("active_methods:", ", ".join(active_methods))

    for dataset_name in datasets:
        print(f"\n=== dataset={dataset_name} model={model_name} ===")
        time_cache, overhead_stages = build_time_cache(project_root, dataset_name, model_name, cost_param)
        base_acc, base_time, base_f1 = no_optional_baseline(project_root, dataset_name, model_name)
        print(f"baseline_no_optional: acc={base_acc:.4f}, time={base_time:.4f}, f1={base_f1:.4f}")

        for budget in budgets:
            print(f"\n--- budget={budget} ---")
            job_builders = {
                "CBAprep": (
                    lambda: cbaprep_search(
                        dataset_name,
                        model_name,
                        budget,
                        args.max_samples,
                        args.max_combos,
                        cost_param,
                        args.optimizer,
                        args.search_mode,
                        benefit_model,
                        args.random_state,
                        args.cbaprep_restarts,
                        not args.no_validation_rerank,
                        args.cbaprep_budget_guard,
                    )
                ),
                "Random-Budget Search": (
                    lambda: random_budget_search(
                        time_cache,
                        overhead_stages,
                        dataset_name,
                        model_name,
                        budget,
                        args.random_samples,
                        args.e_min,
                        cost_param,
                        rng,
                        benefit_model,
                    )
                ),
                "Greedy-Knapsack": (
                    lambda: greedy_knapsack(
                        time_cache,
                        overhead_stages,
                        dataset_name,
                        model_name,
                        budget,
                        args.e_min,
                        e_grid,
                        cost_param,
                        benefit_model,
                    )
                ),
                "Uniform-Budget Allocation": (
                    lambda: uniform_budget_allocation(
                        time_cache,
                        overhead_stages,
                        dataset_name,
                        model_name,
                        budget,
                        args.max_combos,
                        args.e_min,
                        cost_param,
                        rng,
                        random_sample=not args.ordered,
                        benefit_model=benefit_model,
                    )
                ),
            }
            jobs = [(method_name, job_builders[method_name]) for method_name in active_methods]

            for method_name, builder in jobs:
                print(f"[run] {method_name}")
                t0 = time.perf_counter()
                plan = builder()
                search_time = time.perf_counter() - t0
                validation = validate_plan(
                    plan,
                    dataset_name,
                    model_name,
                    repeats=args.execution_repeats,
                )
                add_record(
                    records,
                    dataset_name,
                    model_name,
                    budget,
                    method_name,
                    plan,
                    validation,
                    base_acc,
                    benefit_model=benefit_model,
                )
                records[-1]["search_wall_time"] = float(search_time)
                print(
                    f"  actual_acc={records[-1]['actual_acc']:.4f}, "
                    f"actual_time={records[-1]['actual_time']:.4f}, "
                    f"budget_utilization={records[-1]['budget_utilization']:.4f}, "
                    f"gain_per_cost={records[-1]['gain_per_cost']:.4f}"
                )

    df = pd.DataFrame(records)
    output_suffix = f"{model_name}_{cost_param}"
    if benefit_model != "specific":
        output_suffix = f"{output_suffix}_{benefit_model}"
    csv_path = os.path.join(out_dir, f"budget_aware_baselines_{output_suffix}.csv")
    df.to_csv(csv_path, index=False)

    plots = []
    for dataset_name in datasets:
        part = df[df["dataset"] == dataset_name]
        for metric in ["actual_acc", "actual_time", "budget_utilization", "gain_per_cost"]:
            plots.append(plot_metric(part, metric, out_dir, dataset_name, methods=active_methods))

    print(f"\nSaved: {csv_path}")
    for path in plots:
        print(f"Saved: {path}")
    print("\n=== Summary ===")
    display_cols = [
        "dataset",
        "method",
        "budget",
        "actual_acc",
        "actual_time",
        "budget_utilization",
        "gain_per_cost",
        "selection_acc",
        "valid_evaluated_candidates",
        "pred_acc",
        "pred_time",
    ]
    print(df[display_cols].to_string(index=False))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Budget-aware baseline comparison.")
    parser.add_argument("--datasets", type=str, default="google")
    parser.add_argument("--model", type=str, default="LR")
    parser.add_argument("--budgets", type=str, default="0.3,0.5,0.7,1.0")
    parser.add_argument("--cost_param", type=str, default="e_dataset_method", choices=["e", "e_dataset", "e_dataset_method"])
    parser.add_argument("--benefit_model", type=str, default="specific", choices=["specific", "generalized"])
    parser.add_argument("--max_samples", type=int, default=600)
    parser.add_argument("--max_combos", type=int, default=120)
    parser.add_argument("--random_samples", type=int, default=120)
    parser.add_argument("--optimizer", type=str, default="zofw", choices=["mc", "zofw", "fw", "hybrid_fw"])
    parser.add_argument("--search_mode", type=str, default="optimize_all", choices=["hybrid", "optimize_all"])
    parser.add_argument("--e_min", type=float, default=0.1)
    parser.add_argument("--e_grid", type=str, default="0.1,0.3,0.5,0.7,1.0")
    parser.add_argument("--ordered", action="store_true")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--execution_repeats",
        type=int,
        default=3,
        help="Repeated final executions; median time and predictive metrics are reported.",
    )
    parser.add_argument(
        "--cbaprep_restarts",
        type=int,
        default=3,
        help="Independent CBAPrep discrete-search starts before validation reranking.",
    )
    parser.add_argument(
        "--cbaprep_budget_guard",
        type=float,
        default=0.95,
        help="Validation-time safety factor applied to the execution budget.",
    )
    parser.add_argument(
        "--no_validation_rerank",
        action="store_true",
        help="Select CBAPrep only by GBDT prediction instead of validation reranking.",
    )
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated methods to run. Supports cbaprep, random, greedy, uniform.")
    parser.add_argument("--budget_baselines_only", action="store_true", help="Run only Random-Budget Search, Greedy-Knapsack, and Uniform-Budget Allocation.")
    run_experiment(parser.parse_args())
