import time

import numpy as np

from modules.optimizer.cost_benefit_distribute_fw import FrankWolfeCostOptimizer
from modules.benefit_predictor.acc_predictor import acc


class HybridFrankWolfeCostOptimizer(FrankWolfeCostOptimizer):
    def __init__(self, config):
        super().__init__(config)
        self.n_starts = int(config.get("n_starts", 6))
        self.start_pool_factor = int(config.get("start_pool_factor", 4))
        self.max_iter_per_start = int(
            config.get("max_iter_per_start", max(8, self.max_iter // max(1, self.n_starts)))
        )
        self.grad_avg_samples = int(config.get("grad_avg_samples", 3))
        self.local_refine_steps = int(config.get("local_refine_steps", 24))
        self.local_radius = float(config.get("local_radius", 0.15))
        self.grad_shrink = float(config.get("grad_shrink", 0.5))

    def _e_min_vector(self):
        return np.array([meta["e_min"] for meta in self.variable_stages], dtype=float)

    def _random_feasible_point(self):
        if not self.variable_stages:
            return np.array([], dtype=float)

        e_min = self._e_min_vector()
        time_min = self._time_from_vector(e_min)
        if time_min > self.T_MAX + 1e-12:
            return None

        proposal = np.array(
            [np.random.uniform(meta["e_min"], meta["e_max"]) for meta in self.variable_stages],
            dtype=float,
        )
        proposal = self._project_box(proposal)
        time_prop = self._time_from_vector(proposal)
        if time_prop <= self.T_MAX + 1e-12:
            return proposal

        if time_prop <= time_min + 1e-12:
            return e_min

        alpha = (self.T_MAX - time_min) / max(time_prop - time_min, 1e-12)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        feasible = e_min + alpha * (proposal - e_min)
        return self._project_box(feasible)

    def _project_feasible(self, e_vec):
        e_vec = self._project_box(e_vec)
        total_time = self._time_from_vector(e_vec)
        if total_time <= self.T_MAX + 1e-12:
            return e_vec

        e_min = self._e_min_vector()
        time_min = self._time_from_vector(e_min)
        if time_min > self.T_MAX + 1e-12:
            return None
        if total_time <= time_min + 1e-12:
            return e_min

        alpha = (self.T_MAX - time_min) / max(total_time - time_min, 1e-12)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        feasible = e_min + alpha * (e_vec - e_min)
        return self._project_box(feasible)

    def _select_start_points(self):
        starts = []
        first = self._initial_feasible_point()
        if first is None:
            return []
        starts.append(first)

        target_pool = max(self.n_starts, self.n_starts * self.start_pool_factor)
        attempts = max(10, target_pool * 3)
        for _ in range(attempts):
            candidate = self._random_feasible_point()
            if candidate is None:
                continue
            if any(np.allclose(candidate, s, atol=1e-6, rtol=0.0) for s in starts):
                continue
            starts.append(candidate)
            if len(starts) >= target_pool:
                break

        scored = []
        for s in starts:
            score, _, _ = self._evaluate(s)
            scored.append((score, s.copy()))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [vec for _, vec in scored[: self.n_starts]]

    def _finite_difference_gradient(self, e_vec):
        if len(e_vec) == 0:
            return np.array([], dtype=float)

        grad = np.zeros_like(e_vec)
        base_score, _, _ = self._evaluate(e_vec)
        if not np.isfinite(base_score):
            return grad

        for idx, meta in enumerate(self.variable_stages):
            lo = meta["e_min"]
            hi = meta["e_max"]
            eps_values = []
            for k in range(self.grad_avg_samples):
                factor = self.grad_shrink ** k
                eps_values.append(max(self.grad_eps * factor, 1e-3))

            estimates = []
            for eps in eps_values:
                left = max(lo, e_vec[idx] - eps)
                right = min(hi, e_vec[idx] + eps)
                if right - left <= 1e-12:
                    continue

                x_left = e_vec.copy()
                x_right = e_vec.copy()
                x_left[idx] = left
                x_right[idx] = right

                score_left, _, _ = self._evaluate(x_left)
                score_right, _, _ = self._evaluate(x_right)

                if np.isfinite(score_left) and np.isfinite(score_right):
                    estimates.append((score_right - score_left) / (right - left))
                elif np.isfinite(score_right):
                    estimates.append((score_right - base_score) / max(right - e_vec[idx], 1e-12))
                elif np.isfinite(score_left):
                    estimates.append((base_score - score_left) / max(e_vec[idx] - left, 1e-12))

            if estimates:
                grad[idx] = float(np.median(estimates))
            else:
                grad[idx] = 0.0
        return grad

    def _run_fw_from_start(self, start_vec):
        e_cur = self._project_feasible(start_vec)
        if e_cur is None:
            return None

        best_local_vec = e_cur.copy()
        best_local_score, _, _ = self._evaluate(e_cur)

        for iteration in range(self.max_iter_per_start):
            grad = self._finite_difference_gradient(e_cur)
            s_vec = self._linear_oracle(grad)
            if s_vec is None:
                break

            fw_gap = float(np.dot(grad, s_vec - e_cur))
            if fw_gap <= self.tol:
                break

            e_next = self._feasible_fw_update(e_cur, s_vec, iteration)
            if e_next is None:
                break
            next_score, _, _ = self._evaluate(e_next)
            if np.allclose(e_next, e_cur, atol=self.tol, rtol=0.0):
                break
            e_cur = e_next
            if np.isfinite(next_score) and next_score > best_local_score:
                best_local_score = next_score
                best_local_vec = e_cur.copy()

        return best_local_vec, best_local_score

    def _local_refine(self, center_vec):
        best_vec = center_vec.copy()
        best_score, _, _ = self._evaluate(best_vec)
        if not np.isfinite(best_score):
            return center_vec

        base_scale = np.array(
            [max(meta["e_max"] - meta["e_min"], 1e-6) for meta in self.variable_stages],
            dtype=float,
        )

        for step in range(self.local_refine_steps):
            radius = self.local_radius * (0.96 ** step)
            noise = np.random.normal(loc=0.0, scale=radius, size=len(best_vec)) * base_scale
            candidate = best_vec + noise
            candidate = self._project_feasible(candidate)
            if candidate is None:
                continue
            score, _, _ = self._evaluate(candidate)
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_vec = candidate
        return best_vec

    def optimize(self):
        started_at = time.perf_counter()
        starts = self._select_start_points()
        if not starts:
            if self.warn_infeasible:
                print("Warning: Lower-bound feasible point exceeds total_time.")
            self.optimal.update({
                "config": self.best_time["config"],
                "accuracy": float("nan"),
                "time_used": self.best_time["time_used"],
            })
            result = self._finish(started_at)
            result["optimizer"] = "hybrid_fw"
            return result

        best_vec = None
        best_score = -np.inf
        for start_vec in starts:
            result = self._run_fw_from_start(start_vec)
            if result is None:
                continue
            local_vec, local_score = result
            if np.isfinite(local_score) and local_score > best_score:
                best_score = local_score
                best_vec = local_vec.copy()

        if best_vec is not None:
            refined_vec = self._local_refine(best_vec)
            refined_score, refined_time, refined_cfg = self._evaluate(refined_vec)
            if np.isfinite(refined_score) and refined_score >= self.optimal["accuracy"]:
                self.optimal.update({
                    "config": refined_cfg.copy(),
                    "accuracy": refined_score,
                    "time_used": refined_time,
                })

        if self.optimal["accuracy"] == -np.inf:
            fallback_vec = starts[0]
            fallback_cfg = self._cfg_from_vector(fallback_vec)
            fallback_time = self._calculate_total(fallback_cfg)
            if fallback_time <= self.T_MAX + 1e-12:
                fallback_acc = float(
                    acc(
                        fallback_cfg,
                        model_name=self.model_name,
                        verbose=False,
                        benefit_model=self.benefit_model,
                    )
                )
                self.optimal.update({
                    "config": fallback_cfg,
                    "accuracy": fallback_acc,
                    "time_used": fallback_time,
                })
            else:
                if self.warn_infeasible:
                    print("Warning: No feasible hybrid Frank-Wolfe iterate under total_time. Returning best_time sample.")
                self.optimal.update({
                    "config": self.best_time["config"],
                    "accuracy": float("nan"),
                    "time_used": self.best_time["time_used"],
                })
        result = self._finish(started_at)
        result["optimizer"] = "hybrid_fw"
        return result
