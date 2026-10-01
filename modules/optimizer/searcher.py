import time
import itertools
import numpy as np
import pandas as pd
import os
import sys
import io
import contextlib

from modules.cost_calculator.cost_calculator import (
    build_time_coeff_config,
    Cost,
    build_cost_param_payload,
    get_effective_time_coeff,
    get_dataset_cost_params,
    stage_machine_time,
    enrich_config_by_cost_param,
)
from modules.benefit_predictor.acc_predictor import acc
from modules.optimizer.cost_benefit_distribute import MonteCarloCostOptimizer
from modules.optimizer.cost_benefit_distribute_fw import ZerothOrderFrankWolfeCostOptimizer
from modules.optimizer.cost_benefit_distribute_deterministic_fw import (
    DeterministicFrankWolfeCostOptimizer,
)
from modules.optimizer.cost_benefit_distribute_hybrid_fw import HybridFrankWolfeCostOptimizer
from modules.benefit_predictor.predict_data_generation.run import (
    find_project_root,
    apply_feature_encoding,
    apply_imputation,
    apply_normalization,
    apply_outlier_detection,
    apply_feature_extraction,
    apply_feature_selection,
    apply_duplicate_detection,
)
from modules.methods.preprocessing.data_conversion.data_converter import DataConverter
from modules.methods.preprocessing.dataset_division.dataset_divider import DatasetDivider
from modules.ml_modules.Classification.classifier import Classifier
from modules.common.dataset_utils import (
    DATASET_SHORT_NAMES,
    load_dataset_label,
    resolve_dataset_name,
    resolve_dataset_paths,
)
from modules.common.operator_space import get_method_options, normalize_operator_space


def _predict_time_from_config(time_cfg, e_cfg):
    total_time = 0.0
    for stage in time_cfg['stages']:
        stage_name = stage['name']
        e_key = {
            'duplicate_detection': 'e_duplicate_detection',
            'feature_encoding': 'e_feature_encoding',
            'imputation': 'e_imputation',
            'data_normalization': 'e_data_normalization',
            'outlier_detection': 'e_outlier_detection',
            'feature_extraction': 'e_feature_extraction',
            'feature_selection': 'e_feature_selection',
            'data_conversion': 'e_data_conversion',
            'dataset_division': 'e_dataset_division',
        }.get(stage_name)
        e_val = 0.0 if e_key is None else e_cfg.get(e_key, 0.0)
        total_time += stage_machine_time(stage, e_val)
    return total_time


def _build_pred_config(methods, e_values):
    cfg = {}
    if 'duplicate_detection' in methods and methods['duplicate_detection'] is not None:
        cfg['method_duplicate_detection'] = methods['duplicate_detection']
        cfg['e_duplicate_detection'] = e_values.get('duplicate_detection', 0.0)
    if 'feature_encoding' in methods and methods['feature_encoding'] is not None:
        cfg['method_feature_encoding'] = methods['feature_encoding']
        cfg['e_feature_encoding'] = e_values.get('feature_encoding', 0.0)
    if 'imputation' in methods and methods['imputation'] is not None:
        cfg['method_imputation'] = methods['imputation']
        cfg['e_imputation'] = e_values.get('imputation', 0.0)
    if 'data_normalization' in methods and methods['data_normalization'] is not None:
        cfg['method_data_normalization'] = methods['data_normalization']
        cfg['e_data_normalization'] = e_values.get('data_normalization', 0.0)
    if 'outlier_detection' in methods and methods['outlier_detection'] is not None:
        cfg['method_outlier_detection'] = methods['outlier_detection']
        cfg['e_outlier_detection'] = e_values.get('outlier_detection', 0.0)
    if 'feature_extraction' in methods and methods['feature_extraction'] is not None:
        cfg['method_feature_extraction'] = methods['feature_extraction']
        cfg['e_feature_extraction'] = e_values.get('feature_extraction', 0.0)
    if 'feature_selection' in methods and methods['feature_selection'] is not None:
        cfg['method_feature_selection'] = methods['feature_selection']
        cfg['e_feature_selection'] = e_values.get('feature_selection', 0.0)
    return cfg


def _build_optimizer(time_cfg, optimizer_name):
    optimizer_name = (optimizer_name or "mc").lower()
    if optimizer_name == "hybrid_fw":
        return HybridFrankWolfeCostOptimizer(time_cfg)
    if optimizer_name == "det_fw":
        return DeterministicFrankWolfeCostOptimizer(time_cfg)
    if optimizer_name in {"fw", "zofw"}:
        return ZerothOrderFrankWolfeCostOptimizer(time_cfg)
    if optimizer_name == "mc":
        return MonteCarloCostOptimizer(time_cfg)
    raise ValueError(
        f"Unknown optimizer: {optimizer_name}. "
        "Expected 'mc', 'det_fw', 'zofw' (or compatibility alias 'fw'), "
        "or 'hybrid_fw'."
    )


def _fmt_metric(value):
    if value is None:
        return "None"
    try:
        if isinstance(value, float) and np.isnan(value):
            return "nan"
    except Exception:
        pass
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def _run_actual_pipeline(methods, e_values, model_name='LDA', dataset_name='google', target_name=None, eval_split='test'):
    if eval_split not in {'test', 'val'}:
        raise ValueError("eval_split must be 'test' or 'val'.")
    project_root = find_project_root(os.path.dirname(__file__))
    dataset_name = resolve_dataset_name(dataset_name)
    _, data_path, _ = resolve_dataset_paths(project_root, dataset_name)
    _, target_col = load_dataset_label(project_root, dataset_name)
    if target_name:
        target_col = target_name
    with contextlib.redirect_stdout(io.StringIO()):
        raw = pd.read_csv(data_path)

        t_init = time.perf_counter()

        dc = DataConverter(raw)
        data, _ = dc.transform()
        t_conversion = time.perf_counter()
        dd = DatasetDivider(data, test_rate=0.3, val_rate=0.3)
        dataset_train, dataset_val, dataset_test, _ = dd.transform()
        t_division = time.perf_counter()

        t0 = t_division

        if methods.get('duplicate_detection'):
            dataset_train, dataset_val, dataset_test = apply_duplicate_detection(
                dataset_train, dataset_val, dataset_test,
                target_col, methods['duplicate_detection'], e_values.get('duplicate_detection', 1.0)
            )
        t_duplicate = time.perf_counter()

        # feature encoding / drop non-numeric
        if methods.get('feature_encoding') == 'drop_non_numeric':
            dataset_train = dataset_train.select_dtypes(include=[np.number]).join(dataset_train[target_col])
            dataset_val = dataset_val.select_dtypes(include=[np.number]).join(dataset_val[target_col])
            dataset_test = dataset_test.select_dtypes(include=[np.number]).join(dataset_test[target_col])
        elif methods.get('feature_encoding'):
            dataset_train, dataset_val, dataset_test = apply_feature_encoding(
                dataset_train, dataset_val, dataset_test,
                target_col, methods['feature_encoding'], e_values.get('feature_encoding', 1.0)
            )
        t1 = time.perf_counter()

        if methods.get('imputation'):
            dataset_train, dataset_val, dataset_test = apply_imputation(
                dataset_train, dataset_val, dataset_test,
                target_col, methods['imputation'], e_values.get('imputation', 1.0)
            )
        t2 = time.perf_counter()

        if methods.get('data_normalization'):
            dataset_train, dataset_val, dataset_test = apply_normalization(
                dataset_train, dataset_val, dataset_test,
                target_col, methods['data_normalization'], e_values.get('data_normalization', 1.0)
            )
        t3 = time.perf_counter()

        if methods.get('outlier_detection'):
            dataset_train, dataset_val, dataset_test = apply_outlier_detection(
                dataset_train, dataset_val, dataset_test,
                target_col, methods['outlier_detection'], e_values.get('outlier_detection', 1.0)
            )
        t4 = time.perf_counter()

        if methods.get('feature_extraction'):
            dataset_train, dataset_val, dataset_test = apply_feature_extraction(
                dataset_train, dataset_val, dataset_test,
                target_col, methods['feature_extraction'], e_values.get('feature_extraction', 1.0)
            )
        t5 = time.perf_counter()

        if methods.get('feature_selection'):
            dataset_train, dataset_val, dataset_test = apply_feature_selection(
                dataset_train, dataset_val, dataset_test,
                target_col, methods['feature_selection'], e_values.get('feature_selection', 1.0)
            )
        t6 = time.perf_counter()

        X_train = dataset_train.drop(columns=[target_col])
        y_train = dataset_train[target_col]
        X_val = dataset_val.drop(columns=[target_col])
        y_val = dataset_val[target_col]
        X_eval = dataset_val.drop(columns=[target_col]) if eval_split == 'val' else dataset_test.drop(columns=[target_col])
        y_eval = dataset_val[target_col] if eval_split == 'val' else dataset_test[target_col]
        dataset = {
            'train': X_train.join(y_train),
            'test': X_eval.join(y_eval),
            'target': y_train,
            'target_test': y_eval
        }
        cl = Classifier(dataset, target=target_col, strategy=model_name, k_folds=10, verbose=False)
        metrics = cl.transform(return_metrics=True)
        accuracy = metrics.get("accuracy", None)
        f1 = metrics.get("f1", None)

    actual_total = t6 - t_init
    step_times = {
        'data_conversion': t_conversion - t_init,
        'dataset_division': t_division - t_conversion,
        'duplicate_detection': t_duplicate - t0,
        'feature_encoding': t1 - t_duplicate,
        'imputation': t2 - t1,
        'data_normalization': t3 - t2,
        'outlier_detection': t4 - t3,
        'feature_extraction': t5 - t4,
        'feature_selection': t6 - t5,
    }
    data_stats = {
        'train_rows': int(X_train.shape[0]),
        'train_features': int(X_train.shape[1]),
        'val_rows': int(X_val.shape[0]),
        'val_features': int(X_val.shape[1]),
        'test_rows': int(dataset_test.drop(columns=[target_col]).shape[0]),
        'test_features': int(dataset_test.drop(columns=[target_col]).shape[1]),
        'eval_split': eval_split,
        'eval_rows': int(X_eval.shape[0]),
        'eval_features': int(X_eval.shape[1]),
    }
    return accuracy, f1, actual_total, step_times, data_stats


def search_best_pipeline(
    total_time,
    max_samples,
    model_name='LDA',
    max_combos=200,
    validate=True,
    random_sample=True,
    dataset_name='google',
    target_name=None,
    search_mode='hybrid',
    allow_infeasible=False,
    optimizer_name='mc',
    cost_param='e',
    cost_coeff_mode='scaled',
    runtime_model_version=None,
    runtime_cost_mode='mean',
    benefit_model='specific',
    operator_space='legacy',
    artifact_version=None,
    optional_stages=False,
    zofw_max_iter=20,
    zofw_directions=3,
    zofw_mu=0.10,
    zofw_starts=2,
    zofw_gap_tol=1e-3,
    fw_max_iter=200,
    fw_grad_eps=0.02,
    fw_line_search_points=21,
    fw_gap_tol=1e-4,
    random_state=42,
    objective_mode='budget_performance',
    performance_target=None,
    target_tolerance=1e-6,
    target_penalty=100.0,
    ratio_epsilon=1e-6,
    eval_split='test',
    excluded_method_combinations=None,
):
    dataset_name = resolve_dataset_name(dataset_name)
    operator_space = normalize_operator_space(operator_space)
    objective_mode = ZerothOrderFrankWolfeCostOptimizer.OBJECTIVE_ALIASES.get(
        str(objective_mode).lower()
    )
    if objective_mode is None:
        raise ValueError(
            "objective_mode must be budget_performance, target_cost, or utility_cost_ratio."
        )
    if objective_mode != 'budget_performance' and optimizer_name.lower() not in {'zofw', 'fw'}:
        raise ValueError(
            f"objective_mode='{objective_mode}' requires --optimizer zofw."
        )
    if objective_mode == 'target_cost' and performance_target is None:
        raise ValueError("performance_target is required for objective_mode='target_cost'.")
    method_options = get_method_options(
        operator_space,
        include_optional=optional_stages,
    )
    if optional_stages:
        dataset_params = get_dataset_cost_params(dataset_name)
        if int(dataset_params.get('categorical_cols', 0) or 0) > 0:
            method_options['feature_encoding'] = [
                method for method in method_options['feature_encoding'] if method is not None
            ]
        if float(dataset_params.get('missing_rate', 0.0) or 0.0) > 0:
            method_options['imputation'] = [
                method for method in method_options['imputation'] if method is not None
            ]

    stage_names = list(method_options.keys())
    all_combos = list(itertools.product(*(method_options[s] for s in stage_names)))
    excluded_signatures = set()
    for excluded in excluded_method_combinations or []:
        if isinstance(excluded, dict):
            signature = tuple(excluded.get(stage) for stage in stage_names)
        else:
            signature = tuple(excluded)
        if len(signature) == len(stage_names):
            excluded_signatures.add(signature)
    rng = np.random.default_rng(random_state)
    if random_sample and max_combos < len(all_combos):
        combos = rng.choice(len(all_combos), size=max_combos, replace=False)
        combos = [all_combos[i] for i in combos]
    else:
        combos = all_combos[:max_combos]
    if excluded_signatures:
        combos = [combo for combo in combos if tuple(combo) not in excluded_signatures]

    best = {
        'methods': None,
        'e_values': None,
        'pred_time': float('inf'),
        'pred_acc': -np.inf,
        'surrogate_queries': 0,
        'optimization_time': 0.0,
        'optimizer': optimizer_name,
        'objective_mode': objective_mode,
        'objective_value': float('inf') if objective_mode == 'target_cost' else -np.inf,
        'performance_target': performance_target,
        'target_achieved': False if objective_mode == 'target_cost' else None,
    }
    total_surrogate_queries = 0
    total_resource_optimization_time = 0.0
    optimized_combinations = 0
    direct_utility_evaluations = 0

    def candidate_is_better(candidate_result, pred_acc, pred_time):
        if objective_mode == 'target_cost':
            achieved = bool(candidate_result.get('target_achieved', False))
            current_achieved = bool(best.get('target_achieved', False))
            if achieved != current_achieved:
                return achieved
            if achieved:
                return pred_time < best['pred_time']
            return pred_acc > best['pred_acc']
        if objective_mode == 'utility_cost_ratio':
            value = candidate_result.get('objective_value', -np.inf)
            return value is not None and value > best['objective_value']
        return pred_acc > best['pred_acc']

    def update_best(methods, pred_cfg, pred_acc, pred_time, candidate_result):
        best.update({
            'methods': methods,
            'e_values': {
                stage: pred_cfg.get(f'e_{stage}', 0.0)
                for stage in stage_names
            },
            'pred_time': pred_time,
            'pred_acc': pred_acc,
            'surrogate_queries': candidate_result.get('surrogate_queries', 0),
            'optimization_time': candidate_result.get('optimization_time', 0.0),
            'optimizer': candidate_result.get('optimizer', optimizer_name),
            'objective_mode': objective_mode,
            'objective_value': candidate_result.get('objective_value'),
            'performance_target': performance_target,
            'target_achieved': candidate_result.get('target_achieved'),
            'budget_feasible': candidate_result.get('budget_feasible'),
        })

    tested = 0
    total_to_test = min(max_combos, len(combos))
    last_print = 0
    for combo in combos:
        tested += 1
        if tested == 1 or tested == total_to_test or tested - last_print >= 10:
            pct = tested / total_to_test * 100 if total_to_test > 0 else 0
            bar_len = 30
            filled = int(bar_len * tested / total_to_test) if total_to_test > 0 else 0
            bar = "#" * filled + "-" * (bar_len - filled)
            sys.stdout.write(f"\r[{bar}] {tested}/{total_to_test} ({pct:.1f}%)")
            sys.stdout.flush()
            last_print = tested
        methods = dict(zip(stage_names, combo))


        # build time coeff config with cache
        if tested == 1:
            time_cache = {}
            for cache_stage, opts in method_options.items():
                for m in opts:
                    if m is not None:
                        cfg = build_time_coeff_config(
                            methods={cache_stage: m},
                            total_time=total_time,
                            max_samples=max_samples,
                            model=model_name,
                            verbose=False,
                            dataset=dataset_name,
                            cost_param=cost_param,
                            cost_coeff_mode=cost_coeff_mode,
                            runtime_model_version=runtime_model_version,
                            runtime_cost_mode=runtime_cost_mode,
                        )
                        time_cache[(cache_stage, m)] = cfg['stages'][0]
            for cache_stage, m in {
                'data_conversion': 'data_conversion',
                'dataset_division': 'dataset_division',
            }.items():
                cfg = build_time_coeff_config(
                    methods={cache_stage: m},
                    total_time=total_time,
                    max_samples=max_samples,
                    model=model_name,
                    verbose=False,
                    dataset=dataset_name,
                    cost_param=cost_param,
                    cost_coeff_mode=cost_coeff_mode,
                    runtime_model_version=runtime_model_version,
                    runtime_cost_mode=runtime_cost_mode,
                )
                time_cache[(cache_stage, m)] = cfg['stages'][0]

        methods_for_time = {
            k: v for k, v in methods.items() if v is not None
        }
        methods_for_time['data_conversion'] = 'data_conversion'
        methods_for_time['dataset_division'] = 'dataset_division'

        stages = []
        for stage_name, method in methods_for_time.items():
            stage_template = time_cache.get((stage_name, method), {})
            payload = build_cost_param_payload(stage_name, method, cost_param=cost_param, dataset_name=dataset_name)
            if runtime_model_version:
                effective_time_coeff = stage_template.get('effective_time_coeff', [0.0, 0.0])
            else:
                effective_time_coeff = get_effective_time_coeff(
                    stage_template.get('time_coeff', [0.0, 0.0]),
                    payload,
                    coeff_mode=cost_coeff_mode,
                )
            stages.append({
                'name': stage_name,
                'method': method,
                'e_range': (0.1, 1.0),
                'time_coeff': stage_template.get('time_coeff', [0.0, 0.0]),
                'cost_param_payload': payload,
                'effective_time_coeff': effective_time_coeff,
            })
        time_cfg = {
            'stages': stages,
            'total_time': total_time,
            'max_samples': max_samples,
            'model': model_name,
            'benefit_model': benefit_model,
            'cost_param': cost_param,
            'cost_coeff_mode': cost_coeff_mode,
            'runtime_model_version': runtime_model_version,
            'runtime_cost_mode': runtime_cost_mode,
            'operator_space': operator_space,
            'artifact_version': artifact_version,
            'zofw_max_iter': zofw_max_iter,
            'zofw_directions': zofw_directions,
            'zofw_mu': zofw_mu,
            'zofw_starts': zofw_starts,
            'zofw_gap_tol': zofw_gap_tol,
            'fw_max_iter': fw_max_iter,
            'fw_grad_eps': fw_grad_eps,
            'fw_line_search_points': fw_line_search_points,
            'fw_gap_tol': fw_gap_tol,
            'random_state': random_state,
            'objective_mode': objective_mode,
            'performance_target': performance_target,
            'target_tolerance': target_tolerance,
            'target_penalty': target_penalty,
            'ratio_epsilon': ratio_epsilon,
        }

        if search_mode == 'optimize_all' or objective_mode != 'budget_performance':
            time_cfg['dataset'] = dataset_name
            # During global search, many method combinations are expected to be infeasible.
            # Keep optimizer-level warnings silent and let searcher handle feasibility globally.
            time_cfg['warn_infeasible'] = False
            optimizer = _build_optimizer(time_cfg, optimizer_name)
            result = optimizer.optimize()
            total_surrogate_queries += int(result.get('surrogate_queries', 0) or 0)
            total_resource_optimization_time += float(
                result.get('optimization_time', 0.0) or 0.0
            )
            optimized_combinations += 1
            if result.get('config'):
                pred_cfg = result['config']
                pred_cfg['dataset'] = dataset_name
                pred_cfg['model'] = model_name
                pred_cfg['benefit_model'] = benefit_model
                pred_cfg['operator_space'] = operator_space
                pred_cfg['artifact_version'] = artifact_version
                pred_acc = result.get('accuracy', -np.inf)
                pred_time = result.get('time_used', float('inf'))
                if pred_acc is None or (isinstance(pred_acc, float) and np.isnan(pred_acc)):
                    pred_acc = -np.inf
                if candidate_is_better(result, pred_acc, pred_time):
                    update_best(methods, pred_cfg, pred_acc, pred_time, result)
        else:
            # original hybrid mode
            e_values = {k: 1.0 for k in stage_names}
            pred_cfg = _build_pred_config(methods, e_values)
            pred_cfg['dataset'] = dataset_name
            pred_cfg['model'] = model_name
            pred_cfg['benefit_model'] = benefit_model
            pred_cfg['operator_space'] = operator_space
            pred_cfg['artifact_version'] = artifact_version
            pred_cfg = enrich_config_by_cost_param(
                pred_cfg,
                stages=time_cfg['stages'],
                cost_param=cost_param,
                dataset_name=dataset_name,
            )
            pred_time = _predict_time_from_config(time_cfg, pred_cfg)

            if pred_time <= total_time:
                pred_acc = acc(pred_cfg, model_name=model_name, verbose=False, benefit_model=benefit_model)
                total_surrogate_queries += 1
                direct_utility_evaluations += 1
                direct_result = {
                    'objective_value': pred_acc,
                    'surrogate_queries': 1,
                    'optimization_time': 0.0,
                    'optimizer': optimizer_name,
                    'budget_feasible': True,
                }
                if candidate_is_better(direct_result, pred_acc, pred_time):
                    update_best(methods, pred_cfg, pred_acc, pred_time, direct_result)
            else:
                # check e=0.2 feasibility
                e_low = {k: 0.2 for k in stage_names}
                pred_cfg_low = _build_pred_config(methods, e_low)
                pred_cfg_low['dataset'] = dataset_name
                pred_cfg_low['model'] = model_name
                pred_cfg_low['benefit_model'] = benefit_model
                pred_cfg_low['operator_space'] = operator_space
                pred_cfg_low['artifact_version'] = artifact_version
                pred_cfg_low = enrich_config_by_cost_param(
                    pred_cfg_low,
                    stages=time_cfg['stages'],
                    cost_param=cost_param,
                    dataset_name=dataset_name,
                )
                pred_time_low = _predict_time_from_config(time_cfg, pred_cfg_low)
                if pred_time_low <= total_time:
                    time_cfg['dataset'] = dataset_name
                    time_cfg['warn_infeasible'] = False
                    optimizer = _build_optimizer(time_cfg, optimizer_name)
                    # silence predictor inside optimizer
                    result = optimizer.optimize()
                    total_surrogate_queries += int(
                        result.get('surrogate_queries', 0) or 0
                    )
                    total_resource_optimization_time += float(
                        result.get('optimization_time', 0.0) or 0.0
                    )
                    optimized_combinations += 1
                    if result.get('config'):
                        pred_cfg = result['config']
                        pred_cfg['dataset'] = dataset_name
                        pred_cfg['model'] = model_name
                        pred_cfg['benefit_model'] = benefit_model
                        pred_cfg['operator_space'] = operator_space
                        pred_cfg['artifact_version'] = artifact_version
                        pred_acc = result.get('accuracy', -np.inf)
                        pred_time = result.get('time_used', float('inf'))
                        if candidate_is_better(result, pred_acc, pred_time):
                            update_best(methods, pred_cfg, pred_acc, pred_time, result)
                else:
                    continue

    if total_to_test > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if validate and best['methods'] is not None:
        actual_acc, actual_f1, actual_time, step_times, data_stats = _run_actual_pipeline(
            best['methods'], best['e_values'], model_name=model_name,
            dataset_name=dataset_name, target_name=target_name,
            eval_split=eval_split,
        )
        best['actual_acc'] = actual_acc
        best['actual_f1'] = actual_f1
        best['actual_time'] = actual_time
        best['actual_cost'] = actual_time
        best['actual_target_achieved'] = (
            actual_acc + target_tolerance >= performance_target
            if objective_mode == 'target_cost' and actual_acc is not None
            else None
        )
        best['actual_objective_value'] = (
            actual_time
            if objective_mode == 'target_cost'
            else (
                actual_acc / (max(0.0, actual_time) + ratio_epsilon)
                if objective_mode == 'utility_cost_ratio' and actual_acc is not None
                else actual_acc
            )
        )
        best['step_times'] = step_times
        best['data_stats'] = data_stats

    best['predicted_utility'] = best.get('pred_acc')
    best['predicted_cost'] = best.get('pred_time')
    best['total_surrogate_queries'] = int(total_surrogate_queries)
    best['total_resource_optimization_time'] = float(
        total_resource_optimization_time
    )
    best['optimized_combinations'] = int(optimized_combinations)
    best['direct_utility_evaluations'] = int(direct_utility_evaluations)

    target_infeasible = (
        objective_mode == 'target_cost'
        and best['methods'] is not None
        and not best.get('target_achieved', False)
    )
    if (best['methods'] is None or target_infeasible) and not allow_infeasible:
        raise RuntimeError(
            f"No feasible plan found for objective_mode={objective_mode}, "
            f"total_time={total_time}, performance_target={performance_target}. "
            f"Use --allow_infeasible to suppress this error."
        )
    return best


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search best preprocessing pipeline.")
    parser.add_argument("--dataset", type=str, default=None, help="Single dataset alias or comma-separated aliases (e.g. ada,connect,run_or_walk).")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset aliases. Use --list_datasets to view aliases.")
    parser.add_argument("--all_datasets", action="store_true", help="Run all detected datasets under data/<dataset>/ (requires data.csv + info.json).")
    parser.add_argument("--list_datasets", action="store_true", help="List available datasets under data/ and exit.")
    parser.add_argument("--target", type=str, default=None, help="Target column name override.")
    parser.add_argument(
        "--model",
        type=str,
        default="LDA",
        help="Downstream model name (e.g., LR/NB/CART/RF/XGB/MLP/SVM).",
    )
    parser.add_argument("--total_time", type=float, default=1.0, help="Time budget.")
    parser.add_argument("--total_times", type=str, default=None, help="Comma-separated time budgets, e.g. 0.2,0.5,1.0")
    parser.add_argument("--max_samples", type=int, default=10000, help="Optimizer max_samples.")
    parser.add_argument("--max_combos", type=int, default=5184, help="Max combinations to evaluate.")
    parser.add_argument("--search_mode", type=str, default="hybrid", choices=["hybrid", "optimize_all"], help="Search mode: hybrid (default) or optimize_all (always call optimizer for each combo).")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="mc",
        choices=["mc", "det_fw", "zofw", "fw", "hybrid_fw"],
        help=("Continuous e optimizer: mc, det_fw (deterministic Frank-Wolfe), "
              "zofw, fw (zofw compatibility alias), or hybrid_fw."),
    )
    parser.add_argument("--zofw_max_iter", type=int, default=20, help="Maximum ZOFW iterations per start.")
    parser.add_argument("--zofw_directions", type=int, default=3, help="Random two-point directions per ZOFW iteration.")
    parser.add_argument("--zofw_mu", type=float, default=0.10, help="Initial ZOFW smoothing radius.")
    parser.add_argument("--zofw_starts", type=int, default=2, help="Number of feasible ZOFW starts.")
    parser.add_argument("--zofw_gap_tol", type=float, default=1e-3, help="Estimated FW-gap stopping tolerance.")
    parser.add_argument("--fw_max_iter", type=int, default=200)
    parser.add_argument("--fw_grad_eps", type=float, default=0.02)
    parser.add_argument("--fw_line_search_points", type=int, default=21)
    parser.add_argument("--fw_gap_tol", type=float, default=1e-4)
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for combination sampling and resource optimization.")
    parser.add_argument(
        "--objective_mode",
        default="budget_performance",
        choices=["budget_performance", "target_cost", "utility_cost_ratio"],
        help="Resource objective: maximize performance under budget, minimize cost under a target, or maximize utility/cost.",
    )
    parser.add_argument("--performance_target", type=float, default=None, help="Required predicted performance for target_cost mode.")
    parser.add_argument("--target_tolerance", type=float, default=1e-6, help="Tolerance for deciding whether the performance target is met.")
    parser.add_argument("--target_penalty", type=float, default=100.0, help="Shortfall penalty used by ZOFW in target_cost mode.")
    parser.add_argument("--ratio_epsilon", type=float, default=1e-6, help="Positive denominator stabilizer in utility_cost_ratio mode.")
    parser.add_argument("--cost_param", type=str, default="e", choices=["e", "e_dataset", "e_dataset_method"], help="Cost parameter mode: e, e_dataset, or e_dataset_method.")
    parser.add_argument("--cost_coeff_mode", default="scaled", choices=["scaled", "profiled"], help="Apply legacy multipliers or use directly profiled dataset-method coefficients.")
    parser.add_argument("--runtime_model_version", default=None, help="Optional version under artifacts/cost_models/; current model remains the default.")
    parser.add_argument("--runtime_cost_mode", default="mean", choices=["mean", "q90"], help="Use the versioned mean or runtime-variability upper-quantile coefficient.")
    parser.add_argument("--benefit_model", type=str, default="specific", choices=["specific", "generalized"], help="GBDT benefit predictor type.")
    parser.add_argument("--operator_space", type=str, default="legacy", choices=["legacy", "sota"], help="Operator search space.")
    parser.add_argument("--artifact_version", default=None, help="Benefit artifact version, for example sota_v2.")
    parser.add_argument("--optional_stages", action="store_true", help="Allow valid preprocessing stages to be skipped.")
    parser.add_argument("--allow_infeasible", action="store_true", help="Do not raise error when no feasible plan is found under a time budget.")
    parser.add_argument("--no_validate", action="store_true", help="Skip real validation run.")
    parser.add_argument("--eval_split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    project_root = find_project_root(os.path.dirname(__file__))
    data_root = os.path.join(project_root, "data")
    available_datasets = []
    if os.path.isdir(data_root):
        for name in os.listdir(data_root):
            ds_dir = os.path.join(data_root, name)
            if not os.path.isdir(ds_dir):
                continue
            if os.path.exists(os.path.join(ds_dir, "data.csv")) and os.path.exists(os.path.join(ds_dir, "info.json")):
                available_datasets.append(name)
    available_datasets.sort()

    if args.list_datasets:
        print("Supported dataset aliases:")
        for k in sorted(DATASET_SHORT_NAMES.keys()):
            print(f"{k} -> {DATASET_SHORT_NAMES[k]}")
        sys.exit(0)

    available_map = {x.lower(): x for x in available_datasets}

    def _parse_dataset_tokens(text):
        if not text:
            return []
        return [x.strip().lower() for x in str(text).split(",") if x.strip()]

    def _normalize_dataset_tokens(tokens):
        normalized = []
        missing = []
        for token in tokens:
            alias_resolved = resolve_dataset_name(token)
            key = str(alias_resolved).lower()
            if key in available_map:
                normalized.append(available_map[key])
            else:
                missing.append(token)
        return normalized, missing

    if args.all_datasets:
        dataset_list = available_datasets[:]
        missing = []
    elif args.datasets:
        dataset_list, missing = _normalize_dataset_tokens(_parse_dataset_tokens(args.datasets))
    elif args.dataset:
        # also support "--dataset a,b,c" for convenience
        dataset_list, missing = _normalize_dataset_tokens(_parse_dataset_tokens(args.dataset))
    else:
        dataset_list, missing = _normalize_dataset_tokens(["google"])

    if missing:
        raise ValueError(f"Unknown dataset(s): {missing}. Use --list_datasets to see valid names.")

    if args.objective_mode != "budget_performance":
        total_time_list = [float(args.total_time)]
    elif args.total_times:
        total_time_list = [float(x.strip()) for x in args.total_times.split(",") if x.strip()]
    else:
        total_time_list = [float(args.total_time)]

    failed_cases = []
    for dataset_name in dataset_list:
        for total_time_budget in total_time_list:
            displayed_budget = total_time_budget if args.objective_mode == "budget_performance" else "N/A"
            print(
                f"\n=== Dataset: {dataset_name} | mode: {args.search_mode} "
                f"| optimizer: {args.optimizer} | objective: {args.objective_mode} "
                f"| total_time: {displayed_budget} ==="
            )
            try:
                result = search_best_pipeline(
                    total_time=total_time_budget,
                    max_samples=args.max_samples,
                    model_name=args.model.upper(),
                    max_combos=args.max_combos,
                    validate=not args.no_validate,
                    dataset_name=resolve_dataset_name(dataset_name),
                    target_name=args.target,
                    search_mode=args.search_mode,
                    allow_infeasible=args.allow_infeasible,
                    optimizer_name=args.optimizer,
                    cost_param=args.cost_param,
                    cost_coeff_mode=args.cost_coeff_mode,
                    runtime_model_version=args.runtime_model_version,
                    runtime_cost_mode=args.runtime_cost_mode,
                    benefit_model=args.benefit_model,
                    operator_space=args.operator_space,
                    artifact_version=args.artifact_version,
                    optional_stages=args.optional_stages,
                    zofw_max_iter=args.zofw_max_iter,
                    zofw_directions=args.zofw_directions,
                    zofw_mu=args.zofw_mu,
                    zofw_starts=args.zofw_starts,
                    zofw_gap_tol=args.zofw_gap_tol,
                    fw_max_iter=args.fw_max_iter,
                    fw_grad_eps=args.fw_grad_eps,
                    fw_line_search_points=args.fw_line_search_points,
                    fw_gap_tol=args.fw_gap_tol,
                    random_state=args.random_state,
                    objective_mode=args.objective_mode,
                    performance_target=args.performance_target,
                    target_tolerance=args.target_tolerance,
                    target_penalty=args.target_penalty,
                    ratio_epsilon=args.ratio_epsilon,
                    eval_split=args.eval_split,
                )
            except RuntimeError as exc:
                failed_cases.append((dataset_name, total_time_budget, str(exc)))
                print(f"No feasible plan for dataset={dataset_name}, total_time={total_time_budget}: {exc}")
                continue

            print("\n=== Best Result ===")
            print("dataset:", dataset_name)
            print("total_time:", displayed_budget)
            print("methods:", result['methods'])
            print("e_values:", result['e_values'])
            print(f"pred_time: {_fmt_metric(result.get('pred_time'))}")
            print(f"pred_acc: {_fmt_metric(result.get('pred_acc'))}")
            print(f"objective_mode: {result.get('objective_mode')}")
            print(f"objective_value: {_fmt_metric(result.get('objective_value'))}")
            if result.get('target_achieved') is not None:
                print(f"target_achieved: {result.get('target_achieved')}")
            print(f"surrogate_queries: {result.get('surrogate_queries', 0)}")
            print(f"optimization_time: {_fmt_metric(result.get('optimization_time'))}")
            if result.get('actual_time') is not None:
                print(f"actual_time: {_fmt_metric(result.get('actual_time'))}")
                print(f"actual_acc: {_fmt_metric(result.get('actual_acc'))}")
                if result.get('actual_f1') is not None:
                    print(f"actual_f1: {_fmt_metric(result.get('actual_f1'))}")
                if result.get('step_times'):
                    print("step_times:", result['step_times'])
                if result.get('data_stats'):
                    print("data_stats:", result['data_stats'])

    if failed_cases:
        print("\n=== Infeasible Cases ===")
        for dataset_name, total_time_budget, message in failed_cases:
            print(f"dataset={dataset_name}, total_time={total_time_budget}: {message}")
