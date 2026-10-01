import numpy as np

from modules.optimizer.cost_benefit_distribute_fw import (
    ZerothOrderFrankWolfeCostOptimizer,
)


class DeterministicFrankWolfeCostOptimizer(ZerothOrderFrankWolfeCostOptimizer):
    """Deterministic coordinate finite-difference Frank-Wolfe optimizer.

    This class restores the pre-ZOFW CARO-FW search strategy as an independent
    baseline. It reuses the same cost model, feasible region, fixed operator
    combination, and batched GBDT utility predictor as CARO-ZOFW so that only
    the optimization strategy changes.
    """

    def __init__(self, config):
        fw_config = dict(config)
        fw_config["zofw_max_iter"] = int(
            config.get("fw_max_iter", config.get("max_iter", 200))
        )
        fw_config["zofw_starts"] = 1
        fw_config["zofw_gap_tol"] = float(
            config.get("fw_gap_tol", config.get("tol", 1e-4))
        )
        fw_config["zofw_gap_patience"] = int(config.get("fw_gap_patience", 2))
        fw_config["zofw_min_iter"] = int(config.get("fw_min_iter", 3))
        super().__init__(fw_config)

        self.grad_eps = max(1e-6, float(config.get("fw_grad_eps", 0.02)))
        self.line_search_points = max(
            2, int(config.get("fw_line_search_points", 21))
        )

    def _start_points(self):
        point = self._initial_feasible_point()
        return [] if point is None else [point]

    def _zero_order_gradient(self, e_vec, iteration):
        """Estimate every coordinate derivative with deterministic probes."""
        del iteration
        dimension = len(e_vec)
        if dimension == 0:
            return np.asarray([], dtype=float)

        ranges = self._range_vector()
        probe_pairs = []
        vectors = []
        for index in range(dimension):
            step = self.grad_eps * ranges[index]
            plus_target = np.asarray(e_vec, dtype=float).copy()
            minus_target = np.asarray(e_vec, dtype=float).copy()
            plus_target[index] += step
            minus_target[index] -= step
            plus = self._segment_to_feasible(e_vec, self._project_box(plus_target))
            minus = self._segment_to_feasible(e_vec, self._project_box(minus_target))
            if plus is None or minus is None:
                probe_pairs.append(None)
                continue
            denominator = float(plus[index] - minus[index])
            if abs(denominator) <= 1e-12:
                probe_pairs.append(None)
                continue
            probe_pairs.append((len(vectors), denominator))
            vectors.extend([plus, minus])

        if not vectors:
            return np.zeros(dimension, dtype=float)

        results = self._evaluate_many(vectors)
        gradient = np.zeros(dimension, dtype=float)
        for index, pair in enumerate(probe_pairs):
            if pair is None:
                continue
            offset, denominator = pair
            score_plus = results[offset][0]
            score_minus = results[offset + 1][0]
            if np.isfinite(score_plus) and np.isfinite(score_minus):
                gradient[index] = (score_plus - score_minus) / denominator
        return gradient

    def _feasible_fw_update(self, current, oracle_point, iteration):
        """Choose the best feasible point on the FW segment by grid search."""
        del iteration
        candidates = []
        for gamma in np.linspace(0.0, 1.0, self.line_search_points):
            target = self._project_box(
                (1.0 - float(gamma)) * current + float(gamma) * oracle_point
            )
            candidate = (
                target
                if self._is_feasible(target)
                else self._segment_to_feasible(current, target)
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return None
        results = self._evaluate_many(candidates)
        finite = [
            (float(result[0]), index)
            for index, result in enumerate(results)
            if np.isfinite(result[0])
        ]
        if not finite:
            return None
        _, best_index = max(finite, key=lambda item: (item[0], -item[1]))
        return candidates[best_index]

    def _finish(self, started_at):
        result = super()._finish(started_at)
        result["optimizer"] = "fw"
        return result

    def visualize(self):
        super().visualize()


# Explicit name for scripts that refer to the deterministic CARO-FW baseline.
FrankWolfeCostOptimizer = DeterministicFrankWolfeCostOptimizer
