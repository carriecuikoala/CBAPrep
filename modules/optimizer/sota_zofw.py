import numpy as np

from modules.optimizer.cost_benefit_distribute_fw import (
    ZerothOrderFrankWolfeCostOptimizer,
)


class SOTAZerothOrderFrankWolfeOptimizer(ZerothOrderFrankWolfeCostOptimizer):
    """Apply multi-objective ZOFW to an arbitrary fixed SOTA operator list.

    The base optimizer represents one variable per canonical preprocessing
    stage. SOTA pipelines can contain repeated stages, so this adapter keeps one
    variable per source operator and delegates utility/cost evaluation to the
    experiment's SOTA-aware predictor callbacks.
    """

    def __init__(self, operator_specs, utility_batch_fn, cost_fn, config):
        self.operator_specs = [dict(item) for item in operator_specs]
        self.utility_batch_fn = utility_batch_fn
        self.cost_fn = cost_fn

        adapted = dict(config)
        adapted["stages"] = [
            {
                "name": item["operator_id"],
                "method": item["method"],
                "e_range": tuple(item.get("e_range", (0.0, 1.0))),
                "time_coeff": [0.0, 0.0],
            }
            for item in self.operator_specs
        ]
        super().__init__(adapted)

        # In the SOTA experiment every optimized mode is subject to the same
        # execution budget and performance floor. The objective only determines
        # how feasible allocations are ranked.
        self.use_budget_constraint = bool(adapted.get("enforce_budget", True))
        if self.use_budget_constraint:
            self.T_MAX = float(adapted["total_time"])
        self.enforce_performance_target = bool(
            adapted.get("enforce_performance_target", True)
            and self.performance_target is not None
        )

    def _prepare_stage_metadata(self):
        self.stage_map = {}
        for stage in self.stages:
            e_min, e_max = stage.get("e_range", (0.0, 1.0))
            e_min = float(np.clip(e_min, 0.0, 1.0))
            e_max = float(np.clip(e_max, 0.0, 1.0))
            if e_min > e_max:
                raise ValueError(
                    f"Invalid e_range for operator '{stage['name']}': {(e_min, e_max)}"
                )
            meta = {
                "name": stage["name"],
                "method": stage["method"],
                "m_key": None,
                "e_key": stage["name"],
                "e_min": e_min,
                "e_max": e_max,
                "stage": stage,
            }
            if e_max - e_min > 1e-12:
                self.variable_stages.append(meta)
            else:
                self.fixed_operator_stages.append(meta)

    def _base_config(self):
        return {
            meta["name"]: float(meta["e_min"])
            for meta in self.variable_stages + self.fixed_operator_stages
        }

    def _cfg_from_vector(self, e_vec):
        cfg = self._base_config()
        for index, meta in enumerate(self.variable_stages):
            cfg[meta["name"]] = float(e_vec[index])
        return cfg

    def _vector_from_cfg(self, cfg):
        return np.asarray(
            [float(cfg.get(meta["name"], meta["e_min"])) for meta in self.variable_stages],
            dtype=float,
        )

    def _calculate_total(self, cfg):
        return float(self.cost_fn(dict(cfg)))

    def _target_is_achieved(self, utility):
        if not self.enforce_performance_target:
            return None
        return bool(
            np.isfinite(utility)
            and float(utility) + self.target_tolerance >= self.performance_target
        )

    def _search_value(self, utility, cost):
        value = super()._search_value(utility, cost)
        if (
            self.objective_mode == "utility_cost_ratio"
            and self.enforce_performance_target
            and np.isfinite(utility)
            and np.isfinite(cost)
        ):
            shortfall = max(0.0, self.performance_target - float(utility))
            value -= self.target_penalty * shortfall * shortfall
        return float(value)

    def _consider_candidate(self, cfg, utility, cost):
        if self.objective_mode != "utility_cost_ratio":
            super()._consider_candidate(cfg, utility, cost)
            return
        if not np.isfinite(utility) or not np.isfinite(cost):
            return

        current_utility = self.optimal.get("accuracy", -np.inf)
        current_cost = self.optimal.get("time_used", float("inf"))
        achieved = self._target_is_achieved(utility)
        current_achieved = self._target_is_achieved(current_utility)
        if achieved != current_achieved:
            better = bool(achieved)
        elif achieved or not self.enforce_performance_target:
            better = self._reported_objective_value(
                utility, cost
            ) > self._reported_objective_value(current_utility, current_cost)
        else:
            better = (
                utility > current_utility
                or (
                    np.isclose(utility, current_utility, atol=1e-12, rtol=0.0)
                    and cost < current_cost
                )
            )
        if better:
            self.optimal.update(
                {
                    "config": dict(cfg),
                    "accuracy": float(utility),
                    "time_used": float(cost),
                    "objective_value": self._reported_objective_value(utility, cost),
                    "target_achieved": achieved,
                }
            )

    def _evaluate_many(self, vectors):
        prepared = []
        pending = {}
        for vector in vectors:
            vector = self._project_box(vector)
            cfg = self._cfg_from_vector(vector)
            total_time = self._calculate_total(cfg)
            if total_time < self.best_time["time_used"]:
                self.best_time = {"config": dict(cfg), "time_used": total_time}
            key = self._cache_key(vector)
            prepared.append((key, cfg, total_time))
            violates_budget = (
                self.use_budget_constraint and total_time > self.T_MAX + 1e-9
            )
            if violates_budget or key in self._query_cache:
                continue
            pending.setdefault(key, cfg)

        if pending:
            items = list(pending.items())
            predictions = self.utility_batch_fn([cfg for _, cfg in items])
            for (key, _), prediction in zip(items, predictions):
                score = float(prediction)
                self._query_cache[key] = score
                self.history.append(score)
            self.surrogate_queries += len(items)

        results = []
        for key, cfg, total_time in prepared:
            score = self._query_cache.get(key, -np.inf)
            if self.use_budget_constraint and total_time > self.T_MAX + 1e-9:
                score = -np.inf
            self._consider_candidate(cfg, score, total_time)
            results.append((score, total_time, cfg))
        return results

    def _coordinate_cost_slopes(self):
        lower = self._lower_vector()
        base_cost = self._time_from_vector(lower)
        slopes = []
        for index, meta in enumerate(self.variable_stages):
            width = meta["e_max"] - meta["e_min"]
            if width <= 1e-12:
                slopes.append(0.0)
                continue
            upper = lower.copy()
            upper[index] = meta["e_max"]
            slopes.append(max(0.0, (self._time_from_vector(upper) - base_cost) / width))
        return np.asarray(slopes, dtype=float)
