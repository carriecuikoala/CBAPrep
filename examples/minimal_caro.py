"""Exercise all CARO objectives with synthetic callbacks, not paper results."""
import json
import numpy as np
from modules.optimizer.sota_zofw import SOTAZerothOrderFrankWolfeOptimizer


def utility(configs):
    return np.array([0.6 + 0.15 * np.sqrt(c["scale"]) +
                     0.2 * np.sqrt(c["select"]) for c in configs])


def cost(c):
    return 0.05 + 0.4 * c["scale"] + 0.6 * c["select"]


def main():
    specs = [dict(operator_id="scale", method="min_max", e_range=(0., 1.)),
             dict(operator_id="select", method="VAR", e_range=(0., 1.))]
    for objective in ("budget_performance", "target_cost", "utility_cost_ratio"):
        config = dict(objective_mode=objective, total_time=0.7,
                      performance_target=0.8 if objective == "target_cost" else None,
                      enforce_budget=objective == "budget_performance",
                      enforce_performance_target=objective == "target_cost",
                      ratio_baseline_utility=0.6, random_state=42,
                      zofw_max_iter=20, zofw_directions=3, zofw_starts=2)
        result = SOTAZerothOrderFrankWolfeOptimizer(specs, utility, cost, config).optimize()
        assert result["config"] is not None, result
        assert all(0 <= e <= 1 for e in result["config"].values())
        assert np.isfinite(result["accuracy"])
        if objective == "budget_performance":
            assert result["time_used"] <= 0.7 + 1e-8
        if objective == "target_cost":
            assert result["accuracy"] >= 0.8 - 1e-6
        if objective == "utility_cost_ratio":
            assert result["objective_value"] > (0.95 - 0.6) / 1.05
        print(json.dumps({"objective": objective, "result": result}, default=str))


if __name__ == "__main__":
    main()
