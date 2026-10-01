import argparse
import os
import subprocess
import sys

from modules.common.dataset_utils import find_project_root, resolve_dataset_name
from modules.cost_calculator.extract_dataset_params import get_process_methods
from modules.cost_calculator.cost_calculator import _find_time_file
from modules.benefit_predictor.acc_predictor import get_model_suffix


def _run(cmd, cwd):
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
    )
    current_line = []
    while True:
        ch = proc.stdout.read(1)
        if ch == b"" and proc.poll() is not None:
            break
        if ch == b"":
            continue
        ch = ch.decode("utf-8", errors="replace")
        if ch == "\r":
            if current_line:
                sys.stdout.write("\r" + "".join(current_line))
                sys.stdout.flush()
                current_line = []
            else:
                sys.stdout.write("\r")
                sys.stdout.flush()
            continue
        if ch == "\n":
            sys.stdout.write("".join(current_line) + "\n")
            sys.stdout.flush()
            current_line = []
            continue
        current_line.append(ch)

    if current_line:
        sys.stdout.write("".join(current_line))
        sys.stdout.flush()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def _get_pkl_store_dir(project_root):
    pkl_dir = os.path.join(project_root, "artifacts", "pkl_store")
    os.makedirs(pkl_dir, exist_ok=True)
    return pkl_dir


def _has_all_time_files(dataset_name, step, project_root, cost_param, operator_space):
    missing = []
    found_paths = {}
    for methods in get_process_methods(operator_space).values():
        for method in methods:
            path = _find_time_file(
                method,
                step,
                base_dir=project_root,
                dataset=dataset_name,
                cost_param=cost_param,
                strict_cost_param=(str(cost_param).lower() != "e"),
            )
            if not path:
                missing.append(method)
            else:
                found_paths[method] = path
    return missing, found_paths


def _ensure_time_params(project_root, dataset_name, step, cost_param, operator_space, force_rebuild=False):
    missing, found_paths = _has_all_time_files(
        dataset_name, step, project_root, cost_param, operator_space
    )
    if not missing and not force_rebuild:
        print(f"[skip] time coeff files already exist for dataset={dataset_name}, cost_param={cost_param}")
        sample_paths = list(found_paths.values())[:5]
        for path in sample_paths:
            print(f"  - {path}")
        if len(found_paths) > len(sample_paths):
            print(f"  - ... ({len(found_paths)} files total)")
        return
    if force_rebuild:
        print(f"[force] re-extracting time coeff files for dataset={dataset_name}, cost_param={cost_param}")
    else:
        print(f"[run] extracting time coeff files for dataset={dataset_name}, cost_param={cost_param} (missing {len(missing)} methods)")
    cmd = (
        f"python -m modules.cost_calculator.extract_dataset_params "
        f"--datasets {dataset_name} --step {step} --cost_param {cost_param} "
        f"--operator_space {operator_space} --quiet"
    )
    _run(cmd, project_root)


def _ensure_e_accuracy_dataset(project_root, dataset_name, model_name, iterations, cost_param, operator_space, force_rebuild=False):
    pkl_store_dir = _get_pkl_store_dir(project_root)
    suffix = get_model_suffix(cost_param, operator_space)
    target_path = os.path.join(
        pkl_store_dir,
        f"e_accuracy_dataset_{dataset_name}_{model_name}_{suffix}.pickle",
    )
    if os.path.exists(target_path) and not force_rebuild:
        print(f"[skip] e_accuracy dataset exists: {os.path.basename(target_path)}")
        print(f"  - {target_path}")
        return target_path
    if force_rebuild:
        print(f"[force] regenerating e_accuracy dataset for dataset={dataset_name}, model={model_name}, cost_param={cost_param}")
    else:
        print(f"[run] generating e_accuracy dataset for dataset={dataset_name}, model={model_name}, cost_param={cost_param}")
    cmd = (
        f"python -m modules.benefit_predictor.predict_data_generation.run "
        f"--model {model_name} --dataset {dataset_name} --iterations {iterations} "
        f"--cost_param {cost_param} --operator_space {operator_space} --quiet"
    )
    _run(cmd, project_root)
    return target_path


def _ensure_gbr_model(
    project_root,
    dataset_name,
    model_name,
    cost_param,
    operator_space,
    force_rebuild=False,
    gbr_estimators=100,
    gbr_lr=0.1,
    gbr_depth=3,
    benefit_model="specific",
    dataset_names=None,
    model_names=None,
):
    pkl_store_dir = _get_pkl_store_dir(project_root)
    suffix = get_model_suffix(cost_param, operator_space)
    benefit_model = str(benefit_model or "specific").lower()
    if benefit_model == "generalized":
        gbr_path = os.path.join(pkl_store_dir, "gbr", f"gbr_generalized_{suffix}.pickle")
        scaler_path = os.path.join(pkl_store_dir, "scaler", f"scaler_generalized_{suffix}.pickle")
        train_datasets = ",".join(dataset_names or [dataset_name])
        train_models = ",".join(model_names or [model_name])
    else:
        gbr_path = os.path.join(pkl_store_dir, "gbr", f"gbr_{dataset_name}_{model_name}_{suffix}.pickle")
        scaler_path = os.path.join(pkl_store_dir, "scaler", f"scaler_{dataset_name}_{model_name}_{suffix}.pickle")
        train_datasets = dataset_name
        train_models = model_name
    if os.path.exists(gbr_path) and os.path.exists(scaler_path) and not force_rebuild:
        print(f"[skip] GBR/scaler already exist for benefit_model={benefit_model}, cost_param={cost_param}")
        print(f"  - GBR: {gbr_path}")
        print(f"  - Scaler: {scaler_path}")
        return gbr_path, scaler_path
    if force_rebuild:
        print(f"[force] retraining GBR for benefit_model={benefit_model}, datasets={train_datasets}, models={train_models}, cost_param={cost_param}")
    else:
        print(f"[run] training GBR for benefit_model={benefit_model}, datasets={train_datasets}, models={train_models}, cost_param={cost_param}")
    cmd = (
        f"python -m modules.benefit_predictor.gbr_regressor "
        f"--benefit_model {benefit_model} --models {train_models} --datasets {train_datasets} --cost_param {cost_param} "
        f"--operator_space {operator_space} "
        f"--gbr_estimators {gbr_estimators} --gbr_lr {gbr_lr} --gbr_depth {gbr_depth}"
    )
    _run(cmd, project_root)
    return gbr_path, scaler_path


def _run_search(
    project_root,
    dataset_name,
    model_name,
    optimizer_name,
    cost_param,
    operator_space,
    search_mode,
    total_times,
    max_samples,
    max_combos,
    validate,
    allow_infeasible,
    benefit_model,
    objective_mode,
    performance_target,
    target_tolerance,
    target_penalty,
    ratio_epsilon,
):
    cmd = [
        "python -m modules.optimizer.searcher",
        f"--dataset {dataset_name}",
        f"--model {model_name}",
        f"--optimizer {optimizer_name}",
        f"--cost_param {cost_param}",
        f"--operator_space {operator_space}",
        f"--search_mode {search_mode}",
        f"--total_times {total_times}",
        f"--max_samples {max_samples}",
        f"--max_combos {max_combos}",
        f"--benefit_model {benefit_model}",
        f"--objective_mode {objective_mode}",
        f"--target_tolerance {target_tolerance}",
        f"--target_penalty {target_penalty}",
        f"--ratio_epsilon {ratio_epsilon}",
    ]
    if performance_target is not None:
        cmd.append(f"--performance_target {performance_target}")
    if not validate:
        cmd.append("--no_validate")
    if allow_infeasible:
        cmd.append("--allow_infeasible")
    _run(" ".join(cmd), project_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto run: cost params -> acc model -> search.")
    parser.add_argument("--dataset", type=str, default="house", help="Dataset name or alias.")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names or aliases.")
    parser.add_argument("--model", type=str, default="LR", help="Downstream model name.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated downstream model names for generalized GBR training.")
    parser.add_argument("--optimizer", type=str, default="mc", choices=["mc", "zofw", "fw", "hybrid_fw"], help="Optimizer used in search; fw is a compatibility alias for zofw.")
    parser.add_argument("--cost_param", type=str, default="e", choices=["e", "e_dataset", "e_dataset_method"], help="Cost parameter mode.")
    parser.add_argument("--operator_space", type=str, default="legacy", choices=["legacy", "sota"], help="Operator search space.")
    parser.add_argument("--benefit_model", type=str, default="specific", choices=["specific", "generalized"], help="GBDT benefit predictor type.")
    parser.add_argument("--objective_mode", default="budget_performance", choices=["budget_performance", "target_cost", "utility_cost_ratio"])
    parser.add_argument("--performance_target", type=float, default=None)
    parser.add_argument("--target_tolerance", type=float, default=1e-6)
    parser.add_argument("--target_penalty", type=float, default=100.0)
    parser.add_argument("--ratio_epsilon", type=float, default=1e-6)
    parser.add_argument("--iterations", type=int, default=300, help="Iterations for e_accuracy dataset generation.")
    parser.add_argument("--step", type=float, default=0.1, help="Step used by parameter extraction.")
    parser.add_argument("--search_mode", type=str, default="optimize_all", choices=["hybrid", "optimize_all"], help="Searcher mode.")
    parser.add_argument("--total_times", type=str, default="0.1,0.3,0.5,0.7,1.0,10", help="Comma-separated B_exec budgets.")
    parser.add_argument("--max_samples", type=int, default=10000, help="Optimizer max_samples/max_iter.")
    parser.add_argument("--max_combos", type=int, default=1000, help="Max combinations for searcher.")
    parser.add_argument("--validate", dest="validate", action="store_false", help="Disable actual pipeline validation during search.")
    parser.set_defaults(validate=True)
    parser.add_argument("--allow_infeasible", action="store_true", help="Allow infeasible search cases.")
    parser.add_argument("--force_rebuild", action="store_true", help="Ignore existing files and rebuild time coeffs, e_accuracy dataset, and GBR model from scratch.")
    parser.add_argument("--gbr_estimators", type=int, default=100, help="Number of boosting stages for GBR.")
    parser.add_argument("--gbr_lr", type=float, default=0.1, help="Learning rate for GBR.")
    parser.add_argument("--gbr_depth", type=int, default=3, help="Max depth of individual regression trees in GBR.")
    args = parser.parse_args()

    project_root = find_project_root(os.path.dirname(__file__))
    if project_root is None:
        raise FileNotFoundError("Cannot locate project root.")

    model_name = args.model.upper()
    if args.models:
        train_model_list = [x.strip().upper() for x in str(args.models).split(",") if x.strip()]
    else:
        train_model_list = [model_name]
    if args.datasets:
        dataset_list = [resolve_dataset_name(x.strip()) for x in str(args.datasets).split(",") if x.strip()]
    else:
        dataset_list = [resolve_dataset_name(args.dataset)]

    total_datasets = len(dataset_list)
    if args.benefit_model == "generalized":
        for idx, dataset_name in enumerate(dataset_list, start=1):
            print(f"\n==============================")
            print(f"Dataset {idx}/{total_datasets}: {dataset_name}")
            print(f"==============================")
            print("=== Step 1/4: Ensure time coefficient files ===")
            _ensure_time_params(project_root, dataset_name, args.step, args.cost_param, args.operator_space, force_rebuild=bool(args.force_rebuild))
            print("=== Step 2/4: Ensure e_accuracy datasets ===")
            for train_model in train_model_list:
                _ensure_e_accuracy_dataset(
                    project_root,
                    dataset_name,
                    train_model,
                    args.iterations,
                    args.cost_param,
                    args.operator_space,
                    force_rebuild=bool(args.force_rebuild),
                )

        print("=== Step 3/4: Ensure generalized accuracy prediction model ===")
        _ensure_gbr_model(
            project_root,
            dataset_list[0],
            model_name,
            args.cost_param,
            args.operator_space,
            force_rebuild=bool(args.force_rebuild),
            gbr_estimators=int(args.gbr_estimators),
            gbr_lr=float(args.gbr_lr),
            gbr_depth=int(args.gbr_depth),
            benefit_model=args.benefit_model,
            dataset_names=dataset_list,
            model_names=train_model_list,
        )

        for idx, dataset_name in enumerate(dataset_list, start=1):
            print(f"\n=== Step 4/4: Run search | dataset {idx}/{total_datasets}: {dataset_name} ===")
            _run_search(
                project_root=project_root,
                dataset_name=dataset_name,
                model_name=model_name,
                optimizer_name=args.optimizer,
                cost_param=args.cost_param,
                operator_space=args.operator_space,
                search_mode=args.search_mode,
                total_times=args.total_times,
                max_samples=args.max_samples,
                max_combos=args.max_combos,
                validate=bool(args.validate),
                allow_infeasible=bool(args.allow_infeasible),
                benefit_model=args.benefit_model,
                objective_mode=args.objective_mode,
                performance_target=args.performance_target,
                target_tolerance=args.target_tolerance,
                target_penalty=args.target_penalty,
                ratio_epsilon=args.ratio_epsilon,
            )
        raise SystemExit(0)

    for idx, dataset_name in enumerate(dataset_list, start=1):
        print(f"\n==============================")
        print(f"Dataset {idx}/{total_datasets}: {dataset_name}")
        print(f"==============================")

        print("=== Step 1/4: Ensure time coefficient files ===")
        _ensure_time_params(project_root, dataset_name, args.step, args.cost_param, args.operator_space, force_rebuild=bool(args.force_rebuild))

        print("=== Step 2/4: Ensure e_accuracy dataset ===")
        _ensure_e_accuracy_dataset(
            project_root,
            dataset_name,
            model_name,
            args.iterations,
            args.cost_param,
            args.operator_space,
            force_rebuild=bool(args.force_rebuild),
        )

        print("=== Step 3/4: Ensure accuracy prediction model ===")
        _ensure_gbr_model(
            project_root,
            dataset_name,
            model_name,
            args.cost_param,
            args.operator_space,
            force_rebuild=bool(args.force_rebuild),
            gbr_estimators=int(args.gbr_estimators),
            gbr_lr=float(args.gbr_lr),
            gbr_depth=int(args.gbr_depth),
            benefit_model=args.benefit_model,
            objective_mode=args.objective_mode,
            performance_target=args.performance_target,
            target_tolerance=args.target_tolerance,
            target_penalty=args.target_penalty,
            ratio_epsilon=args.ratio_epsilon,
        )

        print("=== Step 4/4: Run search ===")
        _run_search(
            project_root=project_root,
            dataset_name=dataset_name,
            model_name=model_name,
            optimizer_name=args.optimizer,
            cost_param=args.cost_param,
            operator_space=args.operator_space,
            search_mode=args.search_mode,
            total_times=args.total_times,
            max_samples=args.max_samples,
            max_combos=args.max_combos,
            validate=bool(args.validate),
            allow_infeasible=bool(args.allow_infeasible),
            benefit_model=args.benefit_model,
        )
