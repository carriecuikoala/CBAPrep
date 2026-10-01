import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from modules.benefit_predictor.acc_predictor import acc_batch
from modules.common.dataset_utils import resolve_dataset_name
from modules.cost_calculator.cost_calculator import (
    build_time_coeff_config,
    enrich_config_by_cost_param,
    stage_machine_time,
)


class ZerothOrderFrankWolfeCostOptimizer:
    """Optimize continuous execution degrees with zeroth-order Frank-Wolfe.

    The downstream GBDT utility predictor is treated as a black box. A small
    number of random, antithetic perturbations estimates a search direction for
    a smoothed utility. The returned point is a first-order stationary candidate
    of that smoothed approximation; no global-optimality claim is made.
    """

    OBJECTIVE_ALIASES = {
        "budget_performance": "budget_performance",
        "performance_under_budget": "budget_performance",
        "max_performance": "budget_performance",
        "target_cost": "target_cost",
        "cost_under_target": "target_cost",
        "min_cost": "target_cost",
        "utility_cost_ratio": "utility_cost_ratio",
        "benefit_cost_ratio": "utility_cost_ratio",
        "ratio": "utility_cost_ratio",
    }

    def __init__(self, config):
        self.stages = config["stages"]
        objective_name = str(config.get("objective_mode", "budget_performance")).lower()
        if objective_name not in self.OBJECTIVE_ALIASES:
            choices = sorted(set(self.OBJECTIVE_ALIASES.values()))
            raise ValueError(f"Unknown objective_mode '{objective_name}'. Expected one of {choices}.")
        self.objective_mode = self.OBJECTIVE_ALIASES[objective_name]
        self.use_budget_constraint = self.objective_mode == "budget_performance"
        raw_budget = config.get("total_time")
        if self.use_budget_constraint and raw_budget is None:
            raise ValueError("total_time is required for objective_mode='budget_performance'.")
        self.T_MAX = float(raw_budget) if self.use_budget_constraint else float("inf")
        if self.use_budget_constraint and (not np.isfinite(self.T_MAX) or self.T_MAX < 0.0):
            raise ValueError("total_time must be a finite non-negative value in budget_performance mode.")

        raw_target = config.get("performance_target")
        if self.objective_mode == "target_cost" and raw_target is None:
            raise ValueError("performance_target is required for objective_mode='target_cost'.")
        self.performance_target = None if raw_target is None else float(raw_target)
        if self.performance_target is not None and not 0.0 <= self.performance_target <= 1.0:
            raise ValueError("performance_target must be within [0, 1].")
        self.target_tolerance = max(0.0, float(config.get("target_tolerance", 1e-6)))
        self.target_penalty = max(0.0, float(config.get("target_penalty", 100.0)))
        self.ratio_epsilon = max(1e-12, float(config.get("ratio_epsilon", 1e-6)))
        self.ratio_baseline_utility = float(config.get("ratio_baseline_utility", 0.0))
        if not np.isfinite(self.ratio_baseline_utility):
            raise ValueError("ratio_baseline_utility must be finite.")
        self.model_name = config.get("model", "LDA")
        self.dataset_name = resolve_dataset_name(config.get("dataset", "google"))
        self.benefit_model = config.get("benefit_model", "specific")
        self.cost_param = str(config.get("cost_param", "e")).lower()
        self.operator_space = str(config.get("operator_space", "legacy")).lower()
        self.artifact_version = config.get("artifact_version")
        self.warn_infeasible = bool(config.get("warn_infeasible", True))

        # A small default query budget is intentional. max_samples remains the
        # MC budget and is only used as an upper bound when max_iter is omitted.
        sample_limit = max(1, int(config.get("max_samples", 20)))
        self.max_iter = max(1, int(config.get("zofw_max_iter", config.get("max_iter", min(20, sample_limit)))))
        self.num_directions = max(1, int(config.get("zofw_directions", config.get("num_directions", 3))))
        self.smoothing_radius = float(config.get("zofw_mu", config.get("smoothing_radius", 0.10)))
        self.grad_eps = self.smoothing_radius
        self.mu_decay = float(config.get("zofw_mu_decay", config.get("mu_decay", 0.25)))
        self.n_starts = max(1, int(config.get("zofw_starts", config.get("n_starts", 2))))
        self.gap_tol = max(0.0, float(config.get("zofw_gap_tol", config.get("tol", 1e-3))))
        self.gap_patience = max(1, int(config.get("zofw_gap_patience", 2)))
        self.min_iter = max(1, int(config.get("zofw_min_iter", 3)))
        self.step_numerator = float(config.get("zofw_step_numerator", 2.0))
        self.step_offset = max(1.0, float(config.get("zofw_step_offset", 3.0)))
        self.random_state = int(config.get("random_state", 42))
        self.rng = np.random.default_rng(self.random_state)

        self.stage_map = {
            "duplicate_detection": ("method_duplicate_detection", "e_duplicate_detection"),
            "feature_encoding": ("method_feature_encoding", "e_feature_encoding"),
            "imputation": ("method_imputation", "e_imputation"),
            "data_normalization": ("method_data_normalization", "e_data_normalization"),
            "outlier_detection": ("method_outlier_detection", "e_outlier_detection"),
            "feature_extraction": ("method_feature_extraction", "e_feature_extraction"),
            "feature_selection": ("method_feature_selection", "e_feature_selection"),
        }

        self.variable_stages = []
        self.fixed_operator_stages = []
        self._prepare_stage_metadata()
        self.history = []
        self.fw_gap_history = []
        self.surrogate_queries = 0
        self.iterations_run = 0
        self._query_cache = {}
        self.optimal = self._empty_result()
        self.best_time = {"config": {}, "time_used": float("inf")}

    def _empty_result(self):
        return {
            "config": {},
            "accuracy": -np.inf,
            "time_used": 0.0,
            "surrogate_queries": 0,
            "optimization_time": 0.0,
            "predicted_utility": -np.inf,
            "predicted_cost": 0.0,
            "actual_accuracy": None,
            "actual_cost": None,
            "estimated_fw_gap": None,
            "iterations": 0,
            "feasible": False,
            "execution_degrees_valid": False,
            "optimizer": "zofw",
            "objective_mode": self.objective_mode,
            "objective_value": None,
            "performance_target": self.performance_target,
            "ratio_baseline_utility": self.ratio_baseline_utility,
            "target_achieved": None,
            "budget": self.T_MAX if self.use_budget_constraint else None,
            "budget_feasible": None,
        }

    def _prepare_stage_metadata(self):
        for stage in self.stages:
            stage_name = stage["name"]
            method = stage.get("method")
            if method is None:
                raise ValueError("Each stage must provide a fixed 'method'.")

            e_min, e_max = stage.get("e_range", (0.1, 1.0))
            e_min = float(np.clip(e_min, 0.0, 1.0))
            e_max = float(np.clip(e_max, 0.0, 1.0))
            if e_min > e_max:
                raise ValueError(f"Invalid e_range for stage '{stage_name}': {(e_min, e_max)}")

            if stage_name not in self.stage_map:
                continue

            m_key, e_key = self.stage_map[stage_name]
            meta = {
                "name": stage_name,
                "method": method,
                "m_key": m_key,
                "e_key": e_key,
                "e_min": e_min,
                "e_max": e_max,
                "stage": stage,
            }
            if e_max - e_min > 1e-12:
                self.variable_stages.append(meta)
            else:
                self.fixed_operator_stages.append(meta)

    def _base_config(self):
        cfg = {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "cost_param": self.cost_param,
            "benefit_model": self.benefit_model,
            "operator_space": self.operator_space,
            "artifact_version": self.artifact_version,
        }
        for meta in self.variable_stages + self.fixed_operator_stages:
            cfg[meta["m_key"]] = meta["method"]
        for meta in self.fixed_operator_stages:
            cfg[meta["e_key"]] = meta["e_min"]
        return enrich_config_by_cost_param(
            cfg,
            stages=self.stages,
            cost_param=self.cost_param,
            dataset_name=self.dataset_name,
        )

    def _cfg_from_vector(self, e_vec):
        cfg = self._base_config()
        for idx, meta in enumerate(self.variable_stages):
            cfg[meta["e_key"]] = float(e_vec[idx])
        return cfg

    def _vector_from_cfg(self, cfg):
        return np.asarray(
            [float(cfg.get(meta["e_key"], meta["e_min"])) for meta in self.variable_stages],
            dtype=float,
        )

    def _calculate_total(self, cfg):
        # Sum every stage exactly once. Global stages stay fixed because they do
        # not have an execution-degree entry in stage_map.
        total_time = 0.0
        for stage in self.stages:
            e_val = 0.0
            mapping = self.stage_map.get(stage["name"])
            if mapping is not None:
                e_val = float(cfg.get(mapping[1], 0.0))
            total_time += stage_machine_time(stage, e_val)
        return float(total_time)

    def _time_from_vector(self, e_vec):
        return self._calculate_total(self._cfg_from_vector(e_vec))

    def _lower_vector(self):
        return np.asarray([meta["e_min"] for meta in self.variable_stages], dtype=float)

    def _upper_vector(self):
        return np.asarray([meta["e_max"] for meta in self.variable_stages], dtype=float)

    def _range_vector(self):
        return np.asarray(
            [max(meta["e_max"] - meta["e_min"], 1e-12) for meta in self.variable_stages],
            dtype=float,
        )

    def _project_box(self, e_vec):
        out = np.asarray(e_vec, dtype=float).copy()
        for idx, meta in enumerate(self.variable_stages):
            out[idx] = np.clip(out[idx], meta["e_min"], meta["e_max"])
        return out

    def _target_is_achieved(self, utility):
        if self.objective_mode != "target_cost":
            return None
        return bool(
            np.isfinite(utility)
            and utility + self.target_tolerance >= self.performance_target
        )

    def _search_value(self, utility, cost):
        """Return the scalar black-box merit maximized by the ZOFW update."""
        if not np.isfinite(utility) or not np.isfinite(cost):
            return -np.inf
        if self.objective_mode == "budget_performance":
            return float(utility)
        if self.objective_mode == "utility_cost_ratio":
            gain = float(utility) - self.ratio_baseline_utility
            return gain / (max(0.0, float(cost)) + self.ratio_epsilon)
        shortfall = max(0.0, self.performance_target - float(utility))
        return -float(cost) - self.target_penalty * shortfall * shortfall

    def _reported_objective_value(self, utility, cost):
        if self.objective_mode == "target_cost":
            return float(cost)
        return self._search_value(utility, cost)

    def _consider_candidate(self, cfg, utility, cost):
        if not np.isfinite(utility) or not np.isfinite(cost):
            return

        current_utility = self.optimal.get("accuracy", -np.inf)
        current_cost = self.optimal.get("time_used", float("inf"))
        better = False

        if self.objective_mode == "target_cost":
            achieved = self._target_is_achieved(utility)
            current_achieved = self._target_is_achieved(current_utility)
            if achieved:
                better = (
                    not current_achieved
                    or cost < current_cost - 1e-12
                    or (
                        np.isclose(cost, current_cost, atol=1e-12, rtol=0.0)
                        and utility > current_utility
                    )
                )
            elif not current_achieved:
                better = (
                    utility > current_utility
                    or (
                        np.isclose(utility, current_utility, atol=1e-12, rtol=0.0)
                        and cost < current_cost
                    )
                )
        elif self.objective_mode == "utility_cost_ratio":
            candidate_value = self._search_value(utility, cost)
            current_value = self._search_value(current_utility, current_cost)
            better = candidate_value > current_value
        else:
            better = utility > current_utility

        if better:
            self.optimal.update(
                {
                    "config": cfg.copy(),
                    "accuracy": float(utility),
                    "time_used": float(cost),
                    "objective_value": self._reported_objective_value(utility, cost),
                    "target_achieved": self._target_is_achieved(utility),
                }
            )

    def _is_feasible(self, e_vec, atol=1e-9):
        if e_vec is None:
            return False
        e_vec = np.asarray(e_vec, dtype=float)
        if len(e_vec) != len(self.variable_stages) or not np.all(np.isfinite(e_vec)):
            return False
        for idx, meta in enumerate(self.variable_stages):
            if e_vec[idx] < meta["e_min"] - atol or e_vec[idx] > meta["e_max"] + atol:
                return False
            if e_vec[idx] < -atol or e_vec[idx] > 1.0 + atol:
                return False
        if not self.use_budget_constraint:
            return True
        return self._time_from_vector(e_vec) <= self.T_MAX + atol

    def _segment_to_feasible(self, anchor, target):
        """Return the farthest feasible point on the anchor-target segment."""
        anchor = self._project_box(anchor)
        target = self._project_box(target)
        if not self._is_feasible(anchor):
            return None
        if self._is_feasible(target):
            return target

        lo, hi = 0.0, 1.0
        for _ in range(55):
            mid = 0.5 * (lo + hi)
            candidate = anchor + mid * (target - anchor)
            if self._is_feasible(candidate):
                lo = mid
            else:
                hi = mid
        return self._project_box(anchor + lo * (target - anchor))

    def _initial_feasible_point(self):
        if not self.variable_stages:
            empty = np.asarray([], dtype=float)
            return empty if self._is_feasible(empty) else None
        lower = self._lower_vector()
        if not self._is_feasible(lower):
            return None
        # Balanced initialization avoids the stage-order bias of sequential
        # budget filling while preserving feasibility.
        return self._segment_to_feasible(lower, self._upper_vector())

    def _random_feasible_point(self):
        lower = self._lower_vector()
        if not self._is_feasible(lower):
            return None
        proposal = np.asarray(
            [self.rng.uniform(meta["e_min"], meta["e_max"]) for meta in self.variable_stages],
            dtype=float,
        )
        return self._segment_to_feasible(lower, proposal)

    def _start_points(self):
        first = self._initial_feasible_point()
        if first is None:
            return []
        starts = [first]
        if not self.use_budget_constraint:
            lower = self._lower_vector()
            if not np.allclose(lower, first, atol=1e-10, rtol=0.0):
                starts.append(lower)
        starts = starts[: self.n_starts]
        attempts = 0
        while len(starts) < self.n_starts and attempts < self.n_starts * 10:
            attempts += 1
            candidate = self._random_feasible_point()
            if candidate is None:
                continue
            if any(np.allclose(candidate, item, atol=1e-10, rtol=0.0) for item in starts):
                continue
            starts.append(candidate)
        return starts

    def _cache_key(self, e_vec):
        return tuple(np.round(np.asarray(e_vec, dtype=float), decimals=12).tolist())

    def _evaluate_many(self, vectors):
        """Evaluate feasible vectors in one GBDT batch while counting candidates."""
        prepared = []
        pending = {}
        for position, vector in enumerate(vectors):
            vector = self._project_box(vector)
            cfg = self._cfg_from_vector(vector)
            total_time = self._calculate_total(cfg)
            if total_time < self.best_time["time_used"]:
                self.best_time = {"config": cfg.copy(), "time_used": total_time}
            key = self._cache_key(vector)
            prepared.append([key, cfg, total_time, -np.inf])
            violates_budget = self.use_budget_constraint and total_time > self.T_MAX + 1e-9
            if violates_budget or key in self._query_cache:
                continue
            pending.setdefault(key, (position, cfg))

        if pending:
            pending_items = list(pending.items())
            predictions = acc_batch(
                [item[1][1] for item in pending_items],
                model_name=self.model_name,
                benefit_model=self.benefit_model,
            )
            for (key, _), prediction in zip(pending_items, predictions):
                score = float(prediction)
                self._query_cache[key] = score
                self.history.append(score)
            self.surrogate_queries += len(pending_items)

        results = []
        for key, cfg, total_time, _ in prepared:
            score = self._query_cache.get(key, -np.inf)
            if self.use_budget_constraint and total_time > self.T_MAX + 1e-9:
                score = -np.inf
            self._consider_candidate(cfg, score, total_time)
            results.append((score, total_time, cfg))
        return results

    def _evaluate(self, e_vec):
        return self._evaluate_many([e_vec])[0]

    def _zero_order_gradient(self, e_vec, iteration):
        """Estimate the smoothed utility gradient with random two-point probes."""
        dimension = len(e_vec)
        if dimension == 0:
            return np.asarray([], dtype=float)

        ranges = self._range_vector()
        radius = max(1e-5, self.smoothing_radius / ((iteration + 1.0) ** self.mu_decay))
        grad_normalized = np.zeros(dimension, dtype=float)
        probe_pairs = []

        for _ in range(self.num_directions):
            direction = self.rng.normal(size=dimension)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                continue
            direction /= norm

            plus_target = self._project_box(e_vec + radius * ranges * direction)
            minus_target = self._project_box(e_vec - radius * ranges * direction)
            plus = self._segment_to_feasible(e_vec, plus_target)
            minus = self._segment_to_feasible(e_vec, minus_target)
            if plus is None or minus is None:
                continue

            normalized_delta = (plus - minus) / ranges
            delta_norm = float(np.linalg.norm(normalized_delta))
            if delta_norm <= 1e-12:
                continue

            probe_pairs.append((plus, minus, normalized_delta, delta_norm))

        if not probe_pairs:
            return np.zeros(dimension, dtype=float)

        probe_results = self._evaluate_many(
            [probe for pair in probe_pairs for probe in pair[:2]]
        )
        valid_directions = 0
        for pair_index, (_, _, normalized_delta, delta_norm) in enumerate(probe_pairs):
            score_plus, cost_plus, _ = probe_results[2 * pair_index]
            score_minus, cost_minus, _ = probe_results[2 * pair_index + 1]
            if not np.isfinite(score_plus) or not np.isfinite(score_minus):
                continue

            probe_direction = normalized_delta / delta_norm
            value_plus = self._search_value(score_plus, cost_plus)
            value_minus = self._search_value(score_minus, cost_minus)
            directional_derivative = (value_plus - value_minus) / delta_norm
            grad_normalized += dimension * directional_derivative * probe_direction
            valid_directions += 1

        if valid_directions == 0:
            return np.zeros(dimension, dtype=float)
        grad_normalized /= float(valid_directions)
        return grad_normalized / ranges

    def _coordinate_cost_slopes(self):
        slopes = []
        for meta in self.variable_stages:
            width = meta["e_max"] - meta["e_min"]
            if width <= 1e-12:
                slopes.append(0.0)
                continue
            low_cost = stage_machine_time(meta["stage"], meta["e_min"])
            high_cost = stage_machine_time(meta["stage"], meta["e_max"])
            slopes.append(max(0.0, float(high_cost - low_cost) / width))
        return np.asarray(slopes, dtype=float)

    def _linear_oracle(self, gradient):
        """Solve the FW linear subproblem by fractional unit-cost utility."""
        if not self.variable_stages:
            return np.asarray([], dtype=float)

        lower = self._lower_vector()
        upper = self._upper_vector()
        if not self._is_feasible(lower):
            return None

        if not self.use_budget_constraint:
            return np.where(np.asarray(gradient) > 0.0, upper, lower)

        slopes = self._coordinate_cost_slopes()
        solution = lower.copy()

        # Positive-utility coordinates with no positive incremental cost are
        # allocated first. Remaining coordinates follow gain per unit cost.
        for idx in range(len(solution)):
            if gradient[idx] > 0.0 and slopes[idx] <= 1e-12:
                solution[idx] = upper[idx]

        remaining = max(0.0, self.T_MAX - self._time_from_vector(solution))
        candidates = [
            (float(gradient[idx] / slopes[idx]), idx)
            for idx in range(len(solution))
            if gradient[idx] > 0.0 and slopes[idx] > 1e-12
        ]
        candidates.sort(key=lambda item: (-item[0], item[1]))

        for _, idx in candidates:
            capacity = upper[idx] - solution[idx]
            if capacity <= 1e-12 or remaining <= 1e-12:
                continue
            extra = min(capacity, remaining / slopes[idx])
            solution[idx] += extra
            remaining -= slopes[idx] * extra

        # The fitted stage cost can contain clipping at zero. This final repair
        # makes the oracle robust to small deviations from the affine estimate.
        return self._segment_to_feasible(lower, solution)

    def _step_size(self, iteration):
        return float(np.clip(self.step_numerator / (iteration + self.step_offset), 0.0, 1.0))

    def _feasible_fw_update(self, current, oracle_point, iteration):
        gamma = self._step_size(iteration)
        candidate = self._project_box((1.0 - gamma) * current + gamma * oracle_point)
        if self._is_feasible(candidate):
            return candidate
        return self._segment_to_feasible(current, candidate)

    def _run_from_start(self, start):
        current = self._project_box(start)
        current_score, _, _ = self._evaluate(current)
        if not np.isfinite(current_score):
            return

        small_gap_count = 0
        for iteration in range(self.max_iter):
            gradient = self._zero_order_gradient(current, iteration)
            oracle_point = self._linear_oracle(gradient)
            if oracle_point is None:
                break

            gap = max(0.0, float(np.dot(gradient, oracle_point - current)))
            self.fw_gap_history.append(gap)
            self.iterations_run += 1

            if gap <= self.gap_tol:
                small_gap_count += 1
            else:
                small_gap_count = 0
            if iteration + 1 >= self.min_iter and small_gap_count >= self.gap_patience:
                break

            candidate = self._feasible_fw_update(current, oracle_point, iteration)
            if candidate is None or np.allclose(candidate, current, atol=1e-12, rtol=0.0):
                if iteration + 1 >= self.min_iter:
                    break
                continue
            current = candidate
            current_score, _, _ = self._evaluate(current)
            if not np.isfinite(current_score):
                break

    def _finish(self, started_at):
        elapsed = float(time.perf_counter() - started_at)
        config = self.optimal.get("config") or {}
        if config:
            vector = self._vector_from_cfg(config)
            valid_degrees = all(
                -1e-9 <= float(value) <= 1.0 + 1e-9 for value in vector
            )
            predicted_cost = self._calculate_total(config)
            budget_feasible = (
                predicted_cost <= self.T_MAX + 1e-9
                if self.use_budget_constraint
                else None
            )
            target_achieved = self._target_is_achieved(self.optimal.get("accuracy"))
            feasible = valid_degrees
            if budget_feasible is not None:
                feasible = feasible and budget_feasible
            if target_achieved is not None:
                feasible = feasible and target_achieved
        else:
            valid_degrees = False
            feasible = False
            budget_feasible = False if self.use_budget_constraint else None
            target_achieved = False if self.objective_mode == "target_cost" else None

        predicted_utility = self.optimal.get("accuracy")
        predicted_cost = self.optimal.get("time_used")
        if config and np.isfinite(predicted_utility) and np.isfinite(predicted_cost):
            objective_value = self._reported_objective_value(
                predicted_utility, predicted_cost
            )
        else:
            objective_value = None

        self.optimal.update(
            {
                "surrogate_queries": int(self.surrogate_queries),
                "optimization_time": elapsed,
                "predicted_utility": predicted_utility,
                "predicted_cost": predicted_cost,
                "estimated_fw_gap": self.fw_gap_history[-1] if self.fw_gap_history else None,
                "iterations": int(self.iterations_run),
                "feasible": bool(feasible),
                "execution_degrees_valid": bool(valid_degrees),
                "optimizer": "zofw",
                "objective_mode": self.objective_mode,
                "objective_value": objective_value,
                "objective_direction": "minimize" if self.objective_mode == "target_cost" else "maximize",
                "performance_target": self.performance_target,
                "target_achieved": target_achieved,
                "performance_shortfall": (
                    max(0.0, self.performance_target - predicted_utility)
                    if self.objective_mode == "target_cost" and np.isfinite(predicted_utility)
                    else None
                ),
                "budget": self.T_MAX if self.use_budget_constraint else None,
                "budget_feasible": budget_feasible,
            }
        )
        return self.optimal

    def optimize(self):
        started_at = time.perf_counter()
        starts = self._start_points()
        if not starts:
            if self.warn_infeasible:
                print("Warning: Lower-bound execution vector exceeds total_time.")
            self.optimal.update(
                {
                    "config": self.best_time["config"],
                    "accuracy": float("nan"),
                    "time_used": self.best_time["time_used"],
                }
            )
            return self._finish(started_at)

        for start in starts:
            self._run_from_start(start)

        if self.optimal["accuracy"] == -np.inf:
            score, total_time, cfg = self._evaluate(starts[0])
            self._consider_candidate(cfg, score, total_time)
        return self._finish(started_at)

    def visualize(self):
        plt.figure(figsize=(8, 4))
        if self.history:
            values = np.asarray(self.history, dtype=float)
            plt.plot(np.arange(len(values)), values, "b-", alpha=0.3)
            plt.plot(np.maximum.accumulate(values), "r-", linewidth=2)
        plt.xlabel("Surrogate query")
        plt.ylabel("Predicted utility")
        plt.title("CARO-ZOFW convergence")
        plt.tight_layout()
        plt.show()

    def report(self):
        print("=" * 40)
        print(f"Objective mode: {self.optimal['objective_mode']}")
        print(f"Objective value: {self.optimal['objective_value']}")
        print(f"Predicted utility: {self.optimal['predicted_utility']:.4f}")
        if self.use_budget_constraint:
            print(f"Predicted cost: {self.optimal['predicted_cost']:.4f}/{self.T_MAX:.4f}")
        else:
            print(f"Predicted cost: {self.optimal['predicted_cost']:.4f}")
        if self.objective_mode == "target_cost":
            print(f"Performance target: {self.performance_target:.4f}")
            print(f"Target achieved: {self.optimal['target_achieved']}")
        print(f"Surrogate queries: {self.optimal['surrogate_queries']}")
        print(f"Optimization time: {self.optimal['optimization_time']:.4f}s")
        print("Config:", self.optimal["config"])


# Keep imports and third-party experiment code that used the former class name
# working while routing the implementation to CARO-ZOFW.
FrankWolfeCostOptimizer = ZerothOrderFrankWolfeCostOptimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize execution degrees with the multi-objective CARO-ZOFW interface."
    )
    parser.add_argument("--model", default="LR")
    parser.add_argument("--dataset", default="google")
    parser.add_argument("--total_time", type=float, default=1.0)
    parser.add_argument(
        "--objective_mode",
        default="budget_performance",
        choices=["budget_performance", "target_cost", "utility_cost_ratio"],
    )
    parser.add_argument("--performance_target", type=float, default=None)
    parser.add_argument("--target_tolerance", type=float, default=1e-6)
    parser.add_argument("--target_penalty", type=float, default=100.0)
    parser.add_argument("--ratio_epsilon", type=float, default=1e-6)
    parser.add_argument("--ratio_baseline_utility", type=float, default=0.0)
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--mu", type=float, default=0.10)
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--gap_tol", type=float, default=1e-3)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--benefit_model", default="specific", choices=["specific", "generalized"])
    parser.add_argument("--operator_space", default="legacy", choices=["legacy", "sota"])
    parser.add_argument("--cost_param", default="e", choices=["e", "e_dataset", "e_dataset_method"])
    args = parser.parse_args()

    methods = {
        "data_conversion": "data_conversion",
        "dataset_division": "dataset_division",
        "feature_encoding": "onehot",
        "imputation": "MEAN",
        "data_normalization": "min_max",
        "outlier_detection": "IQR",
        "feature_extraction": "pca",
        "feature_selection": "Tree",
    }
    if args.operator_space == "sota":
        methods["duplicate_detection"] = "ED"

    config = build_time_coeff_config(
        methods=methods,
        total_time=args.total_time,
        max_samples=args.max_iter,
        model=args.model.upper(),
        dataset=resolve_dataset_name(args.dataset),
        cost_param=args.cost_param,
        verbose=False,
    )
    config.update(
        {
            "benefit_model": args.benefit_model,
            "operator_space": args.operator_space,
            "zofw_max_iter": args.max_iter,
            "zofw_directions": args.directions,
            "zofw_mu": args.mu,
            "zofw_starts": args.starts,
            "zofw_gap_tol": args.gap_tol,
            "random_state": args.random_state,
            "objective_mode": args.objective_mode,
            "performance_target": args.performance_target,
            "target_tolerance": args.target_tolerance,
            "target_penalty": args.target_penalty,
            "ratio_epsilon": args.ratio_epsilon,
            "ratio_baseline_utility": args.ratio_baseline_utility,
        }
    )
    optimizer = ZerothOrderFrankWolfeCostOptimizer(config)
    result = optimizer.optimize()
    optimizer.report()
    if args.plot:
        optimizer.visualize()
    print("Optimal:", result)
