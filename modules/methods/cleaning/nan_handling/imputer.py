#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille

import warnings
import time
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter('ignore', category=ImportWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)


class Imputer:
    """
    nan handling: MEAN, MEDIAN, MF, MICE, KNN, DROP, EM, DT, DUMMY

    parameters
    ----------
    * dataset_train (DataFrame)
        transform train data using sampled imputation

    * dataset_val (DataFrame)
        transform validation data using sampled imputation

    * dataset_test (DataFrame)
        transform test data using sampled imputation

    * method (str, default='DROP')
        strategy of nan handling:
        'MEAN': mean imputation (numeric)
        'MEDIAN': median imputation (numeric)
        'MF': most frequent value (all)
        'MICE': IterativeImputer (numeric)
        'KNN': KNN imputation (numeric)
        'DROP': drop rows with missing values
    """

    def __init__(self, e, dataset_train, dataset_val, dataset_test, method='DROP'):
        self.e = e
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.method = method
        self.verbose = False

        self.means_ = None
        self.medians_ = None
        self.most_frequent_ = None
        self.mice_imputer_ = None
        self.knn_imputer_ = None
        self.dt_imputer_ = None
        self.em_mean_ = None
        self.em_cov_ = None
        self.em_columns_ = None

    def get_params(self, deep=True):
        return {
            'e': self.e,
            'method': self.method,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn(f"Ignore invalid params:{k}")
            else:
                setattr(self, k, v)
        return self

    def _fit_mean(self, df_fit):
        numeric_cols = df_fit.select_dtypes(include='number')
        self.means_ = numeric_cols.mean() if not numeric_cols.empty else None

    def _fit_median(self, df_fit):
        numeric_cols = df_fit.select_dtypes(include='number')
        self.medians_ = numeric_cols.median() if not numeric_cols.empty else None

    def _fit_mf(self, df_fit):
        mf = {}
        for col in df_fit.columns:
            if df_fit[col].isnull().any():
                mf[col] = df_fit[col].value_counts(dropna=True).idxmax()
        self.most_frequent_ = mf

    def mean_imputation(self, df):
        numeric_cols = df.select_dtypes(include='number')
        if numeric_cols.empty:
            return df
        if self.means_ is None:
            self._fit_mean(self.dataset_fit)
        filled_numeric = numeric_cols.fillna(self.means_)
        return df.assign(**filled_numeric).copy()

    def median_imputation(self, df):
        numeric_cols = df.select_dtypes(include='number')
        if numeric_cols.empty:
            return df
        if self.medians_ is None:
            self._fit_median(self.dataset_fit)
        filled_numeric = numeric_cols.fillna(self.medians_)
        return df.assign(**filled_numeric).copy()

    def MF_most_frequent_imputation(self, df):
        if self.most_frequent_ is None:
            self._fit_mf(self.dataset_fit)
        out = df.copy()
        for col, value in self.most_frequent_.items():
            if col in out.columns:
                out[col] = out[col].fillna(value)
                if self.verbose:
                    print("Most frequent value for ", col, "is:", value)
        return out

    def MICE_imputation(self, df):
        numeric_cols = df.select_dtypes(['number'])
        if numeric_cols.empty:
            return df
        if self.mice_imputer_ is None:
            self.mice_imputer_ = IterativeImputer(max_iter=3, random_state=0)
            imputed_values = self.mice_imputer_.fit_transform(numeric_cols)
        else:
            imputed_values = self.mice_imputer_.transform(numeric_cols)

        df_imputed = pd.DataFrame(imputed_values,
                                  columns=numeric_cols.columns,
                                  index=df.index)
        non_numeric = df.select_dtypes(exclude=['number'])
        return pd.concat([df_imputed, non_numeric], axis=1)

    def KNN_imputation(self, df, k=4):
        from sklearn.impute import KNNImputer
        numeric_cols = df.select_dtypes(include=['number']).columns
        X = df[numeric_cols]

        if X.isnull().sum().sum() > 0:
            if self.knn_imputer_ is None:
                self.knn_imputer_ = KNNImputer(n_neighbors=k)
                X_filled = self.knn_imputer_.fit_transform(X.values)
            else:
                X_filled = self.knn_imputer_.transform(X.values)
            X_filled_df = pd.DataFrame(X_filled,
                                       columns=numeric_cols,
                                       index=df.index)
            non_numeric = df.select_dtypes(exclude=['number'])
            out = pd.concat([X_filled_df, non_numeric], axis=1)
        else:
            out = df.copy()
        return out

    def _fit_em(self, df_fit, max_iter=10, tol=1e-5):
        numeric = df_fit.select_dtypes(include=['number']).astype(float)
        self.em_columns_ = numeric.columns.tolist()
        if numeric.empty:
            return

        values = numeric.to_numpy(dtype=float)
        missing = np.isnan(values)
        mean = np.nanmean(values, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        filled = np.where(missing, mean, values)

        for _ in range(max_iter):
            previous = filled.copy()
            cov = np.cov(filled, rowvar=False)
            cov = np.atleast_2d(cov)
            cov += np.eye(cov.shape[0]) * 1e-6
            for row_index, row_missing in enumerate(missing):
                if not row_missing.any():
                    continue
                observed = ~row_missing
                if not observed.any():
                    filled[row_index, row_missing] = mean[row_missing]
                    continue
                sigma_mo = cov[np.ix_(row_missing, observed)]
                sigma_oo = cov[np.ix_(observed, observed)]
                delta = filled[row_index, observed] - mean[observed]
                conditional = mean[row_missing] + sigma_mo @ np.linalg.pinv(sigma_oo) @ delta
                filled[row_index, row_missing] = conditional
            mean = np.mean(filled, axis=0)
            if np.nanmax(np.abs(filled - previous)) < tol:
                break

        self.em_mean_ = mean
        self.em_cov_ = np.atleast_2d(np.cov(filled, rowvar=False))
        self.em_cov_ += np.eye(self.em_cov_.shape[0]) * 1e-6

    def EM_imputation(self, df):
        if self.em_mean_ is None:
            self._fit_em(self.dataset_fit)
        if not self.em_columns_:
            return df

        out = df.copy()
        available = [col for col in self.em_columns_ if col in out.columns]
        if not available:
            return out
        column_indices = [self.em_columns_.index(col) for col in available]
        values = out[available].astype(float).to_numpy()
        missing = np.isnan(values)
        mean = self.em_mean_[column_indices]
        cov = self.em_cov_[np.ix_(column_indices, column_indices)]
        for row_index, row_missing in enumerate(missing):
            if not row_missing.any():
                continue
            observed = ~row_missing
            if not observed.any():
                values[row_index, row_missing] = mean[row_missing]
                continue
            sigma_mo = cov[np.ix_(row_missing, observed)]
            sigma_oo = cov[np.ix_(observed, observed)]
            delta = values[row_index, observed] - mean[observed]
            values[row_index, row_missing] = (
                mean[row_missing] + sigma_mo @ np.linalg.pinv(sigma_oo) @ delta
            )
        out.loc[:, available] = values
        return out

    def DT_imputation(self, df):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if not numeric_cols:
            return df
        numeric = df[numeric_cols]
        if self.dt_imputer_ is None:
            self.dt_imputer_ = IterativeImputer(
                estimator=DecisionTreeRegressor(max_features='sqrt', random_state=0),
                max_iter=5,
                random_state=0,
                skip_complete=True,
            )
            values = self.dt_imputer_.fit_transform(numeric)
        else:
            values = self.dt_imputer_.transform(numeric)
        out = df.copy()
        out.loc[:, numeric_cols] = values
        return out

    @staticmethod
    def dummy_imputation(df):
        out = df.copy()
        numeric_cols = out.select_dtypes(include=['number']).columns
        categorical_cols = out.columns.difference(numeric_cols)
        if len(numeric_cols):
            out.loc[:, numeric_cols] = out[numeric_cols].fillna(-1.0)
        if len(categorical_cols):
            out.loc[:, categorical_cols] = out[categorical_cols].fillna('dummy_category')
        return out

    @staticmethod
    def NaN_drop(df):
        return df.dropna()

    def _apply_sampled_imputation(self, df, method_key, sample_idx):
        sample_idx = df.index.intersection(sample_idx)
        df_sample = df.loc[sample_idx].copy()
        df_rest = df.drop(index=sample_idx).copy()

        if method_key == 'MEAN':
            df_sample = self.mean_imputation(df_sample)
        elif method_key == 'MEDIAN':
            df_sample = self.median_imputation(df_sample)
        elif method_key == 'MF':
            df_sample = self.MF_most_frequent_imputation(df_sample)
        elif method_key == 'MICE':
            df_sample = self.MICE_imputation(df_sample)
        elif method_key == 'KNN':
            df_sample = self.KNN_imputation(df_sample)
        elif method_key == 'EM':
            df_sample = self.EM_imputation(df_sample)
        elif method_key == 'DT':
            df_sample = self.DT_imputation(df_sample)
        elif method_key == 'DUMMY':
            df_sample = self.dummy_imputation(df_sample)
        elif method_key == 'DROP':
            df_sample = self.NaN_drop(df_sample)
        else:
            raise ValueError("Invalid nan handling method")

        df_rest = self.NaN_drop(df_rest)
        return pd.concat([df_sample, df_rest], axis=0).sort_index()

    def transform(self):
        """
        apply nan handling on dataset using 'method'(param)

        :return: df_train, df_val, df_test, using_time
        """
        df_train = self.dataset_train.copy()
        df_val = self.dataset_val.copy()
        df_test = self.dataset_test.copy()
        df_fit = df_train.sample(frac=self.e, random_state=42).copy()
        self.dataset_fit = df_fit

        start_time = time.time()

        print("---carrying out nan handling[{}]---\n".format(self.method))

        method_key = str(self.method).strip()

        if method_key in ['MEAN', 'MEDIAN', 'MF']:
            if df_fit.isnull().sum().sum() > 0:
                if method_key == 'MEAN':
                    self._fit_mean(df_fit)
                elif method_key == 'MEDIAN':
                    self._fit_median(df_fit)
                else:
                    self._fit_mf(df_fit)
        elif method_key == 'EM':
            self._fit_em(df_fit)
        elif method_key == 'DT':
            numeric = df_fit.select_dtypes(include=['number'])
            if not numeric.empty:
                self.dt_imputer_ = IterativeImputer(
                    estimator=DecisionTreeRegressor(max_features='sqrt', random_state=0),
                    max_iter=5,
                    random_state=0,
                    skip_complete=True,
                )
                self.dt_imputer_.fit(numeric)

        sample_idx = df_train.sample(frac=self.e, random_state=42).index
        df_train = self._apply_sampled_imputation(df_train, method_key, sample_idx)

        sample_idx_val = df_val.sample(frac=self.e, random_state=42).index
        sample_idx_test = df_test.sample(frac=self.e, random_state=42).index

        df_val = self._apply_sampled_imputation(df_val, method_key, sample_idx_val)
        df_test = self._apply_sampled_imputation(df_test, method_key, sample_idx_test)

        using_time = time.time() - start_time
        print("---nan handling [{}] is over,using time:{}---\n".format(self.method, using_time))

        return df_train, df_val, df_test, using_time


