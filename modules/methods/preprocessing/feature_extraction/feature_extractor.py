import time
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, IncrementalPCA, KernelPCA
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_regression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


class FeatureExtractor:
    """
    feature extraction: pca, truncated_svd, cfs

    parameters
    ----------
    * dataset_train (DataFrame)
        transform train data using fitted extraction

    * dataset_val (DataFrame)
        transform validation data using fitted extraction

    * dataset_test (DataFrame)
        transform test data using fitted extraction

    * target (str)
        target column name for supervised extraction

    * method (str, default='pca')
        strategy of feature extraction:
        'pca': PCA dimensionality reduction
        'truncated_svd': TruncatedSVD dimensionality reduction
        'cfs': CFS-like selection using SelectKBest
    """

    def __init__(self, e, dataset_train, dataset_val, dataset_test, target, method='pca'):
        self.e = e
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.target = target
        self.method = method
        self.n_components = 0.95
        self.cfs_threshold = 0.3
        self.verbose = False
        self.max_generated_features = 256
        self.max_kernel_fit_rows = 500
        self.random_tree_estimators = 20

    def get_params(self, deep=True):
        return {
            'e': self.e,
            'target': self.target,
            'method': self.method,
            'n_components': self.n_components,
            'cfs_threshold': self.cfs_threshold,
            'verbose': self.verbose,
            'max_generated_features': self.max_generated_features,
            'max_kernel_fit_rows': self.max_kernel_fit_rows,
            'random_tree_estimators': self.random_tree_estimators,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn(f"Ignore invalid params:{k}")
            else:
                setattr(self, k, v)
        return self

    def _split_target(self, df):
        target_col = self.target
        if target_col is None or target_col not in df.columns:
            raise ValueError("Target column not found in dataset_fit.")
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return X, y, target_col

    def pca(self, df_fit):
        numeric_fit = df_fit.select_dtypes(include=np.number)
        if numeric_fit.shape[1] == 0:
            raise ValueError("No numeric columns for PCA.")
        return list(numeric_fit.columns)

    def truncated_svd(self, df_fit):
        numeric_fit = df_fit.select_dtypes(include=np.number)
        if numeric_fit.shape[1] == 0:
            raise ValueError("No numeric columns for TruncatedSVD.")
        return list(numeric_fit.columns)

    def cfs(self, df_fit):
        X, Y, _ = self._split_target(df_fit)
        numeric_fit = X.select_dtypes(include=np.number)
        if numeric_fit.shape[1] == 0:
            raise ValueError("No numeric columns for CFS.")

        scaler = StandardScaler()
        X_fit = pd.DataFrame(
            scaler.fit_transform(numeric_fit),
            columns=numeric_fit.columns,
            index=df_fit.index
        )

        if Y.dtype == 'object':
            selector = SelectKBest(mutual_info_classif, k='all')
        else:
            selector = SelectKBest(f_regression, k='all')

        selector.fit(X_fit, Y)
        scores = selector.scores_

        selected_mask = scores >= self.cfs_threshold * np.max(scores)
        selected_features = numeric_fit.columns[selected_mask].tolist()

        if self.verbose:
            score_df = pd.DataFrame({
                'Feature': numeric_fit.columns,
                'Score': scores,
                'Max%': scores / np.max(scores),
                'Selected': selected_mask
            }).sort_values('Score', ascending=False)
            print("\n[Feature Scores]")
            print(score_df.to_string(index=False))
            print(f"\nThreshold: {self.cfs_threshold:.0%} × max = {np.max(scores) * self.cfs_threshold:.3f}")

        return selected_features

    def _apply_columns(self, df, columns_to_keep, target_col=None):
        df_out = df.copy()
        if target_col is not None and target_col in df_out.columns:
            df_out = df_out.drop(columns=[target_col])
        columns = [c for c in columns_to_keep if c in df_out.columns]
        return df_out[columns]

    def _pca_transform(self, df_fit, df_train, df_val, df_test):
        numeric_fit = df_fit.select_dtypes(include=np.number)
        numeric_train = df_train.select_dtypes(include=np.number)
        numeric_val = df_val.select_dtypes(include=np.number)
        numeric_test = df_test.select_dtypes(include=np.number)

        # Align transform inputs to fit columns:
        # df_fit is a sampled subset, so it may miss one-hot columns that exist in train/val/test.
        fit_cols = list(numeric_fit.columns)
        numeric_train = numeric_train.reindex(columns=fit_cols, fill_value=0.0)
        numeric_val = numeric_val.reindex(columns=fit_cols, fill_value=0.0)
        numeric_test = numeric_test.reindex(columns=fit_cols, fill_value=0.0)

        scaler = StandardScaler()
        X_fit = scaler.fit_transform(numeric_fit)
        X_train = scaler.transform(numeric_train)
        X_val = scaler.transform(numeric_val)
        X_test = scaler.transform(numeric_test)

        if self.verbose:
            corr_matrix = pd.DataFrame(X_fit, columns=numeric_fit.columns).corr().abs()
            upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
            print("\n[PCA Precheck]")
            print(f"Mean correlation: {upper_triangle.mean():.2f}")
            print(f"Max correlation: {upper_triangle.max():.2f}")
            print(f"High-corr pairs (>0.8): {np.sum(upper_triangle > 0.8)}")

        pca = PCA(n_components=self.n_components)
        train_pca = pca.fit_transform(X_fit)
        component_names = [f'PC{i + 1}' for i in range(train_pca.shape[1])]

        df_train_pca = pd.DataFrame(pca.transform(X_train), columns=component_names, index=df_train.index)
        df_val_pca = pd.DataFrame(pca.transform(X_val), columns=component_names, index=df_val.index)
        df_test_pca = pd.DataFrame(pca.transform(X_test), columns=component_names, index=df_test.index)

        return df_train_pca, df_val_pca, df_test_pca

    def _svd_transform(self, df_fit, df_train, df_val, df_test):
        numeric_fit = df_fit.select_dtypes(include=np.number)
        numeric_train = df_train.select_dtypes(include=np.number)
        numeric_val = df_val.select_dtypes(include=np.number)
        numeric_test = df_test.select_dtypes(include=np.number)

        # align columns across splits
        common_cols = (
            numeric_fit.columns
            .intersection(numeric_train.columns)
            .intersection(numeric_val.columns)
            .intersection(numeric_test.columns)
        )
        numeric_fit = numeric_fit[common_cols]
        numeric_train = numeric_train[common_cols]
        numeric_val = numeric_val[common_cols]
        numeric_test = numeric_test[common_cols]

        n_features = numeric_fit.shape[1]
        if n_features < 1:
            raise ValueError("No numeric columns for TruncatedSVD.")

        n_components = self.n_components
        if isinstance(n_components, float):
            if n_components <= 0 or n_components >= 1:
                raise ValueError("n_components as float must be in (0, 1).")
            n_components = int(np.ceil(n_features * n_components))
        n_components = int(n_components)
        if n_components < 1:
            n_components = 1
        if n_features > 1:
            n_components = min(n_components, n_features - 1)
        else:
            n_components = 1

        scaler = StandardScaler(with_mean=False)
        X_fit = scaler.fit_transform(numeric_fit)
        X_train = scaler.transform(numeric_train)
        X_val = scaler.transform(numeric_val)
        X_test = scaler.transform(numeric_test)

        svd = TruncatedSVD(n_components=n_components)
        svd.fit(X_fit)

        train_svd = svd.transform(X_train)
        val_svd = svd.transform(X_val)
        test_svd = svd.transform(X_test)
        component_names = [f'SVD{i + 1}' for i in range(train_svd.shape[1])]

        df_train_svd = pd.DataFrame(train_svd, columns=component_names, index=df_train.index)
        df_val_svd = pd.DataFrame(val_svd, columns=component_names, index=df_val.index)
        df_test_svd = pd.DataFrame(test_svd, columns=component_names, index=df_test.index)

        return df_train_svd, df_val_svd, df_test_svd

    @staticmethod
    def _aligned_numeric(df_fit, df_train, df_val, df_test):
        numeric_fit = df_fit.select_dtypes(include=np.number)
        if numeric_fit.shape[1] == 0:
            raise ValueError("No numeric columns for feature extraction.")
        columns = numeric_fit.columns.tolist()
        return (
            numeric_fit,
            df_train.select_dtypes(include=np.number).reindex(columns=columns, fill_value=0.0),
            df_val.select_dtypes(include=np.number).reindex(columns=columns, fill_value=0.0),
            df_test.select_dtypes(include=np.number).reindex(columns=columns, fill_value=0.0),
        )

    @staticmethod
    def _frames_from_arrays(train, val, test, names, train_index, val_index, test_index):
        return (
            pd.DataFrame(train, columns=names, index=train_index),
            pd.DataFrame(val, columns=names, index=val_index),
            pd.DataFrame(test, columns=names, index=test_index),
        )

    def _polynomial_transform(self, df_fit, df_train, df_val, df_test, interaction_only=False):
        numeric_fit, numeric_train, numeric_val, numeric_test = self._aligned_numeric(
            df_fit, df_train, df_val, df_test
        )
        variances = numeric_fit.var().sort_values(ascending=False)
        max_inputs = max(1, int((np.sqrt(9 + 8 * self.max_generated_features) - 3) // 2))
        columns = variances.index[:max_inputs].tolist()
        numeric_fit = numeric_fit[columns]
        numeric_train = numeric_train[columns]
        numeric_val = numeric_val[columns]
        numeric_test = numeric_test[columns]
        transformer = PolynomialFeatures(
            degree=2,
            interaction_only=interaction_only,
            include_bias=False,
        )
        transformer.fit(numeric_fit)
        names = transformer.get_feature_names_out(columns).tolist()
        return self._frames_from_arrays(
            transformer.transform(numeric_train),
            transformer.transform(numeric_val),
            transformer.transform(numeric_test),
            names,
            df_train.index,
            df_val.index,
            df_test.index,
        )

    def _integer_components(self, n_features, n_samples, strict=False):
        value = self.n_components
        if isinstance(value, float) and 0 < value < 1:
            value = int(np.ceil(n_features * value))
        value = max(1, int(value))
        upper = min(n_features, max(1, n_samples - (1 if strict else 0)), 50)
        if strict:
            upper = min(upper, max(1, min(n_features, n_samples) - 1))
        return min(value, upper)

    def _component_transform(self, df_fit, df_train, df_val, df_test, method):
        numeric_fit, numeric_train, numeric_val, numeric_test = self._aligned_numeric(
            df_fit, df_train, df_val, df_test
        )
        scaler = StandardScaler()
        fit_values = scaler.fit_transform(numeric_fit)
        train_values = scaler.transform(numeric_train)
        val_values = scaler.transform(numeric_val)
        test_values = scaler.transform(numeric_test)

        strict = method in {'pca_arpack'}
        components = self._integer_components(numeric_fit.shape[1], numeric_fit.shape[0], strict=strict)
        if min(numeric_fit.shape) < 2:
            names = [f'{method}_1']
            return self._frames_from_arrays(
                train_values[:, :1], val_values[:, :1], test_values[:, :1], names,
                df_train.index, df_val.index, df_test.index,
            )
        if method == 'incremental_pca':
            transformer = IncrementalPCA(n_components=components)
        elif method == 'kernel_pca':
            if fit_values.shape[0] > self.max_kernel_fit_rows:
                sample = np.random.default_rng(0).choice(
                    fit_values.shape[0],
                    size=self.max_kernel_fit_rows,
                    replace=False,
                )
                fit_values = fit_values[sample]
            transformer = KernelPCA(
                n_components=components,
                kernel='rbf',
                eigen_solver='randomized',
                random_state=0,
            )
        elif method == 'pca_arpack':
            transformer = PCA(n_components=components, svd_solver='arpack', random_state=0)
        elif method == 'pca_randomized':
            transformer = PCA(n_components=components, svd_solver='randomized', random_state=0)
        else:
            raise ValueError(f'Unknown component method: {method}')
        transformer.fit(fit_values)
        names = [f'{method}_{i + 1}' for i in range(components)]
        return self._frames_from_arrays(
            transformer.transform(train_values),
            transformer.transform(val_values),
            transformer.transform(test_values),
            names,
            df_train.index,
            df_val.index,
            df_test.index,
        )

    def _random_trees_transform(self, df_fit, df_train, df_val, df_test):
        numeric_fit, numeric_train, numeric_val, numeric_test = self._aligned_numeric(
            df_fit, df_train, df_val, df_test
        )
        transformer = RandomTreesEmbedding(
            n_estimators=self.random_tree_estimators,
            max_depth=3,
            random_state=0,
            sparse_output=False,
        )
        transformer.fit(numeric_fit)
        train_values = transformer.transform(numeric_train)
        val_values = transformer.transform(numeric_val)
        test_values = transformer.transform(numeric_test)
        names = [f'RTE{i + 1}' for i in range(train_values.shape[1])]
        return self._frames_from_arrays(
            train_values, val_values, test_values, names,
            df_train.index, df_val.index, df_test.index,
        )

    def transform(self):
        """
        apply feature extraction on dataset using 'method'(param)

        :return: df_train, df_val, df_test, using_time
        """
        df_train = self.dataset_train.copy()
        df_val = self.dataset_val.copy()
        df_test = self.dataset_test.copy()
        df_fit = df_train.sample(frac=self.e, random_state=42).copy()

        start_time = time.time()

        print("---carrying out feature extraction[{}]---\n".format(self.method))

        method_key = str(self.method).strip()

        if method_key == 'pca':
            df_train, df_val, df_test = self._pca_transform(
                df_fit.drop(columns=[self.target], errors='ignore'),
                df_train.drop(columns=[self.target], errors='ignore'),
                df_val.drop(columns=[self.target], errors='ignore'),
                df_test.drop(columns=[self.target], errors='ignore'),
            )

        elif method_key in ['truncated_svd', 'svd', 'tsvd']:
            df_train, df_val, df_test = self._svd_transform(
                df_fit.drop(columns=[self.target], errors='ignore'),
                df_train.drop(columns=[self.target], errors='ignore'),
                df_val.drop(columns=[self.target], errors='ignore'),
                df_test.drop(columns=[self.target], errors='ignore'),
            )

        elif method_key == 'cfs':
            to_keep = self.cfs(df_fit)
            df_train = self._apply_columns(df_train, to_keep, self.target)
            df_val = self._apply_columns(df_val, to_keep, self.target)
            df_test = self._apply_columns(df_test, to_keep, self.target)

        elif method_key in ['polynomial', 'interaction']:
            df_train, df_val, df_test = self._polynomial_transform(
                df_fit.drop(columns=[self.target], errors='ignore'),
                df_train.drop(columns=[self.target], errors='ignore'),
                df_val.drop(columns=[self.target], errors='ignore'),
                df_test.drop(columns=[self.target], errors='ignore'),
                interaction_only=(method_key == 'interaction'),
            )

        elif method_key in ['incremental_pca', 'kernel_pca', 'pca_arpack', 'pca_randomized']:
            df_train, df_val, df_test = self._component_transform(
                df_fit.drop(columns=[self.target], errors='ignore'),
                df_train.drop(columns=[self.target], errors='ignore'),
                df_val.drop(columns=[self.target], errors='ignore'),
                df_test.drop(columns=[self.target], errors='ignore'),
                method_key,
            )

        elif method_key == 'random_trees_embedding':
            df_train, df_val, df_test = self._random_trees_transform(
                df_fit.drop(columns=[self.target], errors='ignore'),
                df_train.drop(columns=[self.target], errors='ignore'),
                df_val.drop(columns=[self.target], errors='ignore'),
                df_test.drop(columns=[self.target], errors='ignore'),
            )

        else:
            raise ValueError("Invalid feature extraction method\n")

        using_time = time.time() - start_time
        print("---feature extraction [{}] is over,using time:{}---\n".format(self.method, using_time))

        return df_train, df_val, df_test, using_time


def simple_feature_select_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep a small subset of features and encode non-numeric columns so
    extraction can run on Google Play Store data.
    """
    selected_numeric = ['Rating', 'Reviews', 'Size', 'Installs', 'Price']
    selected_categorical = ['Category', 'Type', 'Content Rating']
    selected_cols = [c for c in selected_numeric + selected_categorical if c in df.columns]
    data = df[selected_cols].copy()

    for col in selected_numeric:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col] = data[col].fillna(data[col].median())

    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    data = data.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    return data


