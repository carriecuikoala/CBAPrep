import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from modules.cost_calculator.cost_calculator import (
    Cost,
    build_time_coeff_config,
    stage_machine_time,
    enrich_config_by_cost_param,
)
from modules.benefit_predictor.acc_predictor import acc_batch
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
from modules.common.dataset_utils import resolve_dataset_paths, load_dataset_label, resolve_dataset_name

import os


machine_time = Cost.machine_time

class MonteCarloCostOptimizer:
    def __init__(self, config):
        """
        优化器配置
        config:
            - stages: list of stage dicts (fixed method per stage)
            - total_time: time constraint
            - max_samples: number of samples
              - model: downstream model name (LDA/CART/NB/MNB/LR)
        """
        self.stages = config['stages']
        self.T_MAX = config['total_time']
        self.model_name = config.get('model', 'LDA')
        self.dataset_name = resolve_dataset_name(config.get('dataset', 'google'))
        self.benefit_model = config.get('benefit_model', 'specific')
        self.max_samples = config.get('max_samples', 100000)
        self.random_state = int(config.get('random_state', 42))
        self.rng = np.random.default_rng(self.random_state)
        self.warn_infeasible = bool(config.get('warn_infeasible', True))
        self.cost_param = str(config.get('cost_param', 'e')).lower()
        self.operator_space = str(config.get('operator_space', 'legacy')).lower()
        self.artifact_version = config.get('artifact_version')

        self.stage_map = {
            'duplicate_detection': ('method_duplicate_detection', 'e_duplicate_detection'),
            'feature_encoding': ('method_feature_encoding', 'e_feature_encoding'),
            'imputation': ('method_imputation', 'e_imputation'),
            'data_normalization': ('method_data_normalization', 'e_data_normalization'),
            'outlier_detection': ('method_outlier_detection', 'e_outlier_detection'),
            'feature_extraction': ('method_feature_extraction', 'e_feature_extraction'),
            'feature_selection': ('method_feature_selection', 'e_feature_selection'),
        }

        self.optimal = {
            'config': {},
            'accuracy': -np.inf,
            'time_used': 0.0,
        }
        self.history = []
        self.surrogate_queries = 0
        self.best_time = {
            'config': {},
            'time_used': float('inf'),
        }

    def _generate_sample(self):
        cfg = {}
        cfg['dataset'] = self.dataset_name
        cfg['model'] = self.model_name
        cfg['cost_param'] = self.cost_param
        cfg['benefit_model'] = self.benefit_model
        cfg['operator_space'] = self.operator_space
        cfg['artifact_version'] = self.artifact_version
        for stage in self.stages:
            method = stage.get('method')
            if method is None:
                raise ValueError("Each stage must provide a fixed 'method'.")
            e_min, e_max = stage.get('e_range', (0.1, 1.0))
            e_val = float(self.rng.uniform(e_min, e_max))
            if stage['name'] in self.stage_map:
                m_key, e_key = self.stage_map[stage['name']]
                cfg[m_key] = method
                cfg[e_key] = e_val
        return enrich_config_by_cost_param(cfg, stages=self.stages, cost_param=self.cost_param, dataset_name=self.dataset_name)

    def _calculate_total(self, cfg):
        total_time = 0.0
        for stage in self.stages:
            stage_name = stage['name']
            e_val = 0.0
            if stage_name in self.stage_map:
                _, e_key = self.stage_map[stage_name]
                e_val = cfg.get(e_key, 0.0)
            total_time += stage_machine_time(stage, e_val)
        return total_time

    def optimize(self):
        started_at = time.perf_counter()
        feasible_configs = []
        feasible_times = []
        for _ in range(self.max_samples):
            cfg = self._generate_sample()
            total_time = self._calculate_total(cfg)
            if total_time < self.best_time['time_used']:
                self.best_time['config'] = cfg.copy()
                self.best_time['time_used'] = total_time
            if total_time <= self.T_MAX:
                feasible_configs.append(cfg.copy())
                feasible_times.append(float(total_time))
        if feasible_configs:
            predictions = acc_batch(
                feasible_configs,
                model_name=self.model_name,
                benefit_model=self.benefit_model,
            )
            # A query denotes one candidate utility evaluation, even though the
            # GBDT implementation evaluates the candidates in a vectorized call.
            self.surrogate_queries += len(feasible_configs)
            self.history.extend(predictions.tolist())
            best_idx = int(np.nanargmax(predictions))
            self.optimal.update({
                'config': feasible_configs[best_idx].copy(),
                'accuracy': float(predictions[best_idx]),
                'time_used': float(feasible_times[best_idx]),
            })
        if self.optimal['accuracy'] == -np.inf:
            if self.warn_infeasible:
                print("Warning: No feasible sample under total_time. Returning best_time sample.")
            self.optimal.update({
                'config': self.best_time['config'],
                'accuracy': float('nan'),
                'time_used': self.best_time['time_used'],
            })
        optimization_time = float(time.perf_counter() - started_at)
        config = self.optimal.get('config') or {}
        degrees = [
            float(config[e_key])
            for _, e_key in self.stage_map.values()
            if e_key in config
        ]
        valid_degrees = bool(config) and all(0.0 <= value <= 1.0 for value in degrees)
        feasible = (
            valid_degrees
            and self.optimal.get('time_used', float('inf')) <= self.T_MAX + 1e-9
        )
        self.optimal.update({
            'surrogate_queries': int(self.surrogate_queries),
            'optimization_time': optimization_time,
            'predicted_utility': self.optimal.get('accuracy'),
            'predicted_cost': self.optimal.get('time_used'),
            'actual_accuracy': None,
            'actual_cost': None,
            'feasible': bool(feasible),
            'execution_degrees_valid': bool(valid_degrees),
            'optimizer': 'mc',
        })
        return self.optimal

    def visualize(self):
        plt.figure(figsize=(8, 4))
        if self.history:
            plt.plot(np.arange(len(self.history)), self.history, 'b-', alpha=0.3)
            plt.plot(np.maximum.accumulate(self.history), 'r-', lw=2)
        plt.xlabel('Sample Index')
        plt.ylabel('Accuracy')
        plt.title('Optimization Convergence')
        plt.tight_layout()
        plt.show()

    def report(self):
        print("=" * 40)
        print(f"Optimal Accuracy: {self.optimal['accuracy']:.4f}")
        print(f"Time Used: {self.optimal['time_used']:.1f}/{self.T_MAX}")
        print(f"Surrogate queries: {self.optimal.get('surrogate_queries', 0)}")
        print(f"Optimization time: {self.optimal.get('optimization_time', 0.0):.4f}s")
        print("Cost Used: N/A")
        print("Config:", self.optimal['config'])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Optimize e for fixed methods under time constraint.")
    parser.add_argument("--model", type=str, default="LDA", help="Downstream model name (e.g., LDA/CART/NB/MNB/LR).")
    parser.add_argument("--dataset", type=str, default="google", help="Dataset alias (e.g., abalone, ada, connect, jungle, run_or_walk).")
    parser.add_argument("--target", type=str, default=None, help="Target column name override.")
    parser.add_argument("--plot", action="store_true", help="Show optimization convergence plot.")
    parser.add_argument("--benefit_model", type=str, default="specific", choices=["specific", "generalized"], help="GBDT benefit predictor type.")
    parser.add_argument("--operator_space", type=str, default="legacy", choices=["legacy", "sota"], help="Operator search space.")
    parser.add_argument("--cost_param", type=str, default="e", choices=["e", "e_dataset", "e_dataset_method"])
    args = parser.parse_args()

    methods = {
        'data_conversion': 'data_conversion',
        'dataset_division': 'dataset_division',
        'feature_encoding': 'onehot',
        'imputation': 'MEAN',
        'data_normalization': 'min_max',
        'outlier_detection': 'IQR',
        'feature_extraction': 'pca',
        'feature_selection': 'Tree',
    }
    if args.operator_space == 'sota':
        methods['duplicate_detection'] = 'ED'
    config = build_time_coeff_config(
        methods=methods,
        total_time=1.0,
        max_samples=100000,
        model=args.model.upper(),
        dataset=resolve_dataset_name(args.dataset.strip().lower()),
        cost_param=args.cost_param,
    )
    config['dataset'] = resolve_dataset_name(args.dataset.strip().lower())
    config['benefit_model'] = args.benefit_model
    config['operator_space'] = args.operator_space
    print(config)

    optimizer = MonteCarloCostOptimizer(config)
    result = optimizer.optimize()
    optimizer.report()
    if args.plot:
        optimizer.visualize()
    print("Optimal:", result)

    # validation: run pipeline with optimal e and evaluate accuracy on LDA
    if result and result.get('config'):
        opt_cfg = result['config']
        time_used = optimizer._calculate_total(opt_cfg)

        project_root = find_project_root(os.path.dirname(__file__))
        dataset_name = resolve_dataset_name(args.dataset.strip().lower())
        _, data_path, _ = resolve_dataset_paths(project_root, dataset_name)
        _, target_col = load_dataset_label(project_root, dataset_name)
        if args.target:
            target_col = args.target

        t_init=time.perf_counter()

        raw = pd.read_csv(data_path)
        dc = DataConverter(raw)
        data, _ = dc.transform()
        dd = DatasetDivider(data, test_rate=0.3, val_rate=0.3)
        dataset_train, dataset_val, dataset_test, _ = dd.transform()

        # target_col set based on dataset_name above
        e_enc = opt_cfg.get('e_feature_encoding', 0.0)
        e_dup = opt_cfg.get('e_duplicate_detection', 0.0)
        e_imp = opt_cfg.get('e_imputation', 0.0)
        e_norm = opt_cfg.get('e_data_normalization', 0.0)
        e_out = opt_cfg.get('e_outlier_detection', 0.0)
        e_ext = opt_cfg.get('e_feature_extraction', 0.0)
        e_sel = opt_cfg.get('e_feature_selection', 0.0)

        t0 = time.perf_counter()
        if opt_cfg.get('method_duplicate_detection'):
            dataset_train, dataset_val, dataset_test = apply_duplicate_detection(
                dataset_train, dataset_val, dataset_test,
                target_col, opt_cfg.get('method_duplicate_detection'), e_dup
            )
        t_duplicate = time.perf_counter()
        dataset_train, dataset_val, dataset_test = apply_feature_encoding(
            dataset_train, dataset_val, dataset_test,
            target_col, opt_cfg.get('method_feature_encoding'), e_enc
        )
        t1 = time.perf_counter()
        dataset_train, dataset_val, dataset_test = apply_imputation(
            dataset_train, dataset_val, dataset_test,
            target_col, opt_cfg.get('method_imputation'), e_imp
        )
        t2 = time.perf_counter()
        dataset_train, dataset_val, dataset_test = apply_normalization(
            dataset_train, dataset_val, dataset_test,
            target_col, opt_cfg.get('method_data_normalization'), e_norm
        )
        t3 = time.perf_counter()
        dataset_train, dataset_val, dataset_test = apply_outlier_detection(
            dataset_train, dataset_val, dataset_test,
            target_col, opt_cfg.get('method_outlier_detection'), e_out
        )
        t4 = time.perf_counter()
        dataset_train, dataset_val, dataset_test = apply_feature_extraction(
            dataset_train, dataset_val, dataset_test,
            target_col, opt_cfg.get('method_feature_extraction'), e_ext
        )
        t5 = time.perf_counter()
        dataset_train, dataset_val, dataset_test = apply_feature_selection(
            dataset_train, dataset_val, dataset_test,
            target_col, opt_cfg.get('method_feature_selection'), e_sel
        )
        t6 = time.perf_counter()

        from modules.ml_modules.Classification.classifier import Classifier
        X_train = dataset_train.drop(columns=[target_col])
        y_train = dataset_train[target_col]
        X_test = dataset_test.drop(columns=[target_col])
        y_test = dataset_test[target_col]
        dataset = {
            'train': X_train.join(y_train),
            'test': X_test.join(y_test),
            'target': y_train,
            'target_test': y_test
        }
        cl = Classifier(dataset, target=target_col, strategy=args.model.upper(), k_folds=10, verbose=False)
        accuracy = cl.transform()

        actual_total_time = t6 - t0
        print("\n=== Validation ===")
        print(f"Time Used (recalc): {t6-t0:.4f}")
        print(f"Accuracy ({args.model.upper()} actual): {accuracy:.4f}")
        print("\n=== Optimizer Prediction ===")
        print(f"Time Used (pred): {result.get('time_used', 0):.4f}")
        print(f"Accuracy (pred): {result.get('accuracy', 0):.4f}")
        print("\n=== Step Times (Actual) ===")
        print(f"duplicate_detection: {t_duplicate - t0:.4f}")
        print(f"feature_encoding: {t1 - t_duplicate:.4f}")
        print(f"imputation: {t2 - t1:.4f}")
        print(f"data_normalization: {t3 - t2:.4f}")
        print(f"outlier_detection: {t4 - t3:.4f}")
        print(f"feature_extraction: {t5 - t4:.4f}")
        print(f"feature_selection: {t6 - t5:.4f}")
        print(f"total_actual_time: {actual_total_time:.4f}")

        print("\n=== Step Times (Predicted) ===")
        for stage in config['stages']:
            stage_name = stage['name']
            e_key = optimizer.stage_map.get(stage_name, (None, None))[1]
            e_val = opt_cfg.get(e_key, 0.0) if e_key else 0.0
            pred_time = stage_machine_time(stage, e_val)
            print(f"{stage_name}: {pred_time:.4f}")
        print(f"total_pred_time: {time_used:.4f}")
        print(f"total_process_time: {t6-t_init:.4f}")
