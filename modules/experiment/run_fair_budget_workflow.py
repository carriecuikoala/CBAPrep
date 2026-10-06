import argparse
import os
import shutil
import traceback
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.common.dataset_utils import DATASET_SHORT_NAMES, find_project_root, resolve_dataset_name
from modules.experiment.budget_aware_baselines import get_active_methods, run_experiment


METRICS = ["actual_acc", "actual_time", "budget_utilization", "gain_per_cost"]


def parse_float_list(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def safe_budget_name(value):
    text = f"{float(value):g}"
    return text.replace(".", "p")


def safe_tag(value):
    text = str(value).strip().lower()
    chars = []
    for ch in text:
        if ch.isalnum():
            chars.append(ch)
        elif ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    tag = "".join(chars).strip("_")
    return tag or "fair"


def output_tag(args, active_methods):
    if getattr(args, "output_tag", None):
        return safe_tag(args.output_tag)
    if active_methods == ["Random-Budget Search", "Greedy-Knapsack", "Uniform-Budget Allocation"]:
        return "budget_only"
    return "fair"


def markdown_table(frame):
    cols = list(frame.columns)
    rows = [[format_cell(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    widths = []
    for idx, col in enumerate(cols):
        widths.append(max([len(str(col))] + [len(row[idx]) for row in rows]))
    header = "| " + " | ".join(str(col).ljust(widths[idx]) for idx, col in enumerate(cols)) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(cols))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(cols))) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def format_cell(value):
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def make_single_args(args, dataset_name, budget):
    return SimpleNamespace(
        datasets=dataset_name,
        model=args.model,
        budgets=str(float(budget)),
        cost_param=args.cost_param,
        benefit_model=args.benefit_model,
        max_samples=args.max_samples,
        max_combos=args.max_combos,
        random_samples=args.random_samples,
        optimizer=args.optimizer,
        search_mode=args.search_mode,
        e_min=args.e_min,
        e_grid=args.e_grid,
        ordered=args.ordered,
        random_state=args.random_state,
        execution_repeats=getattr(args, "execution_repeats", 3),
        cbaprep_restarts=getattr(args, "cbaprep_restarts", 3),
        cbaprep_budget_guard=getattr(args, "cbaprep_budget_guard", 0.95),
        no_validation_rerank=getattr(args, "no_validation_rerank", False),
        methods=args.methods,
        budget_baselines_only=args.budget_baselines_only,
    )


def output_dir(project_root):
    path = os.path.join(project_root, "artifacts", "experiments", "budget", "baselines")
    os.makedirs(path, exist_ok=True)
    return path


def per_budget_path(out_dir, dataset_name, budget, model, cost_param, tag, benefit_model="specific"):
    return os.path.join(
        out_dir,
        f"budget_aware_baselines_{dataset_name}_budget{safe_budget_name(budget)}_{model}_{cost_param}_{benefit_model}_{tag}.csv",
    )


def plot_metric(df, metric, out_dir, dataset_name, model, cost_param, tag, methods):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for method_name in methods:
        part = df[df["method"] == method_name].sort_values("budget")
        if part.empty:
            continue
        ax.plot(part["budget"], part[metric], marker="o", linewidth=1.8, label=method_name)
    ax.set_xlabel("Budget")
    ax.set_ylabel(metric)
    ax.set_title(f"{dataset_name}: {metric} under fair budget-aware baselines")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, f"{tag}_{dataset_name}_{model}_{cost_param}_{metric}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def write_summaries(df, out_dir, model, cost_param, tag, methods, benefit_model="specific"):
    for col in METRICS + ["budget", "selection_acc", "valid_evaluated_candidates"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    combined_path = os.path.join(out_dir, f"budget_aware_baselines_{tag}_{model}_{cost_param}_{benefit_model}.csv")
    df.to_csv(combined_path, index=False)

    avg = (
        df.groupby("method", as_index=False)[METRICS]
        .mean(numeric_only=True)
        .sort_values("actual_acc", ascending=False)
    )
    avg_path = os.path.join(out_dir, f"budget_aware_baselines_{tag}_average_by_method_{model}_{cost_param}_{benefit_model}.csv")
    avg.to_csv(avg_path, index=False)

    best_rows = []
    for (dataset_name, budget), part in df.groupby(["dataset", "budget"]):
        valid = part.dropna(subset=["actual_acc"])
        if valid.empty:
            continue
        row = valid.loc[valid["actual_acc"].idxmax()]
        best_rows.append(
            {
                "dataset": dataset_name,
                "budget": budget,
                "best_method_by_acc": row["method"],
                "best_actual_acc": row["actual_acc"],
                "actual_time": row["actual_time"],
                "budget_utilization": row["budget_utilization"],
                "gain_per_cost": row["gain_per_cost"],
            }
        )
    best = pd.DataFrame(best_rows)
    best_path = os.path.join(out_dir, f"budget_aware_baselines_{tag}_best_by_budget_{model}_{cost_param}_{benefit_model}.csv")
    best.to_csv(best_path, index=False)

    detail_cols = [
        "dataset",
        "budget",
        "method",
        "actual_acc",
        "actual_time",
        "budget_utilization",
        "gain_per_cost",
        "selection_acc",
        "valid_evaluated_candidates",
    ]
    detail_cols = [col for col in detail_cols if col in df.columns]
    detail = df[detail_cols].sort_values(["dataset", "budget", "method"])

    md_path = os.path.join(out_dir, f"budget_aware_baselines_{tag}_result_tables_{model}_{cost_param}_{benefit_model}.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Fair Budget-aware Baselines Results\n\n")
        handle.write("## Detailed Records\n\n")
        handle.write(markdown_table(detail))
        handle.write("\n\n## Average by Method\n\n")
        handle.write(markdown_table(avg))
        handle.write("\n\n## Best Method by Dataset and Budget\n\n")
        handle.write(markdown_table(best))
        handle.write("\n")

    plots = []
    for dataset_name in sorted(df["dataset"].dropna().unique()):
        part = df[df["dataset"] == dataset_name]
        for metric in METRICS:
            plots.append(plot_metric(part, metric, out_dir, dataset_name, model, f"{cost_param}_{benefit_model}", tag, methods))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    avg_for_plot = avg.copy()
    avg_for_plot["method"] = pd.Categorical(avg_for_plot["method"], methods, ordered=True)
    avg_for_plot = avg_for_plot.sort_values("method")
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for ax, metric in zip(axes.ravel(), METRICS):
        ax.bar(avg_for_plot["method"].astype(str), avg_for_plot[metric], color=colors)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    avg_plot = os.path.join(out_dir, f"budget_aware_baselines_{tag}_average_metrics_{model}_{cost_param}_{benefit_model}.png")
    fig.savefig(avg_plot, dpi=300)
    plt.close(fig)
    plots.append(avg_plot)

    if not best.empty:
        counts = best["best_method_by_acc"].value_counts().reindex(methods, fill_value=0)
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        ax.bar(counts.index, counts.values, color=colors)
        ax.set_ylabel("Best-accuracy count")
        ax.set_title("Best method count across dataset-budget cases")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        count_plot = os.path.join(out_dir, f"budget_aware_baselines_{tag}_best_count_{model}_{cost_param}_{benefit_model}.png")
        fig.savefig(count_plot, dpi=300)
        plt.close(fig)
        plots.append(count_plot)

    return {
        "combined": combined_path,
        "average": avg_path,
        "best": best_path,
        "markdown": md_path,
        "plots": plots,
    }


def run_workflow(args):
    project_root = find_project_root(os.path.dirname(__file__))
    if project_root is None:
        raise FileNotFoundError("Cannot locate project root.")
    out_dir = output_dir(project_root)
    if getattr(args, "all_datasets", False):
        dataset_inputs = list(DATASET_SHORT_NAMES.keys())
    else:
        dataset_inputs = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    datasets = [resolve_dataset_name(name) for name in dataset_inputs]
    budgets = parse_float_list(args.budgets)
    model = args.model.upper()
    cost_param = str(args.cost_param).lower()
    benefit_model = str(args.benefit_model).lower()
    active_methods = get_active_methods(args)
    tag = output_tag(args, active_methods)
    completed = []
    failed = []
    print("active_methods:", ", ".join(active_methods))
    print("output_tag:", tag)

    for dataset_name in datasets:
        for budget in budgets:
            target_path = per_budget_path(out_dir, dataset_name, budget, model, cost_param, tag, benefit_model=benefit_model)
            if os.path.exists(target_path) and not args.force:
                print(f"[skip] existing result: {target_path}")
                completed.append(target_path)
                continue

            print(f"\n=== fair run: dataset={dataset_name}, budget={budget}, model={model} ===")
            try:
                run_experiment(make_single_args(args, dataset_name, budget))
                source_suffix = f"{model}_{cost_param}"
                if benefit_model != "specific":
                    source_suffix = f"{source_suffix}_{benefit_model}"
                source_path = os.path.join(out_dir, f"budget_aware_baselines_{source_suffix}.csv")
                if not os.path.exists(source_path):
                    raise FileNotFoundError(f"Missing expected result: {source_path}")
                shutil.copyfile(source_path, target_path)
                completed.append(target_path)
                print(f"[saved] {target_path}")
            except KeyboardInterrupt:
                print("\nInterrupted by user. Completed per-budget files are preserved.")
                raise
            except Exception as exc:
                failed.append({"dataset": dataset_name, "budget": budget, "error": str(exc)})
                print(f"[failed] dataset={dataset_name}, budget={budget}: {exc}")
                traceback.print_exc()
                if not args.continue_on_error:
                    raise

    if not completed:
        raise RuntimeError("No completed per-budget result files were found.")

    frames = []
    for path in completed:
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    combined = pd.concat(frames, ignore_index=True)
    artifacts = write_summaries(combined, out_dir, model, cost_param, tag, active_methods, benefit_model=benefit_model)

    if failed:
        failed_path = os.path.join(out_dir, f"budget_aware_baselines_{tag}_failed_{model}_{cost_param}_{benefit_model}.csv")
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        artifacts["failed"] = failed_path

    print("\n=== Fair workflow outputs ===")
    for key, value in artifacts.items():
        if isinstance(value, list):
            for item in value:
                print(f"{key}: {item}")
        else:
            print(f"{key}: {value}")

    display_cols = [
        "dataset",
        "budget",
        "method",
        "actual_acc",
        "actual_time",
        "budget_utilization",
        "gain_per_cost",
        "selection_acc",
        "valid_evaluated_candidates",
    ]
    display_cols = [col for col in display_cols if col in combined.columns]
    print("\n=== Combined summary ===")
    print(combined[display_cols].to_string(index=False))
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run fair budget-aware baseline experiments one dataset-budget case at a time, "
            "then merge CSV files and generate summary figures."
        )
    )
    parser.add_argument("--datasets", type=str, default="google")
    parser.add_argument(
        "--all_datasets",
        action="store_true",
        help="Run all registered datasets using their command-line aliases.",
    )
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
        help="Repeated final executions; report medians to reduce runtime noise.",
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
        help="Disable validation reranking and use the surrogate-best CBAPrep plan directly.",
    )
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated methods to run. Supports cbaprep, random, greedy, uniform.")
    parser.add_argument("--budget_baselines_only", action="store_true", help="Run only Random-Budget Search, Greedy-Knapsack, and Uniform-Budget Allocation.")
    parser.add_argument("--output_tag", type=str, default=None, help="Optional output filename tag. Defaults to fair or budget_only.")
    parser.add_argument("--force", action="store_true", help="Rerun cases even when per-budget CSV files already exist.")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue remaining cases after a failed dataset-budget run.")
    run_workflow(parser.parse_args())
