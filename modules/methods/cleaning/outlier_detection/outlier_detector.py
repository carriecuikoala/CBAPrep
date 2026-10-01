#!/usr/bin/env python3
# coding: utf-8
# Author:Laure Berti-Equille <laure.berti@ird.fr>

import warnings
import time
import numpy as np
import pandas as pd


class OutlierDetector:
    """
    outlier detection: ZSB, IQR, LOF

    parameters
    ----------
    * dataset_train (DataFrame)
        transform train data using sampled detection

    * dataset_val (DataFrame)
        transform validation data using sampled detection

    * dataset_test (DataFrame)
        transform test data using sampled detection

    * method (str, default='ZSB')
        strategy of outlier detection:
        'ZSB': robust Z-score with MAD
        'IQR': IQR rule
        'LOF': Local Outlier Factor
    """

    def __init__(self, e, dataset_train, dataset_val, dataset_test, method='ZSB'):
        self.e = e
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.method = method
        self.threshold = 0.3
        self.verbose = False
        self.lof_model_ = None
        self.lof_scaler_ = None
        self.mad_median_ = None
        self.mad_scale_ = None
        self.mad_fill_ = None

    def get_params(self, deep=True):
        return {
            'e': self.e,
            'method': self.method,
            'threshold': self.threshold,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn(f"Ignore invalid params:{k}")
            else:
                setattr(self, k, v)
        return self

    @staticmethod
    def IQR_outlier_detection(verbose, dataset, threshold):
        print("IQR_outlier_detection is performing\n")

        X = dataset.select_dtypes(['number'])
        Y = dataset.select_dtypes(['object'])

        if len(X.columns) < 1:
            print("Error: Need at least one numeric variable for IQR detection")
            return dataset

        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1

        outliers = X[((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR)))]
        to_drop = X[outliers.sum(axis=1) / outliers.shape[1] > threshold].index
        to_keep = set(X.index) - set(to_drop)

        if threshold == -1:
            X = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
        else:
            X = X.loc[list(to_keep)]

        df = X.join(Y)

        print(len(to_drop), "outlying rows have been removed")

        if len(to_drop) > 0 and verbose:
            print("with indexes:", list(to_drop))
            print("\nOutliers:\n", dataset.loc[to_drop])

        print("IQR_outlier_detection is over\n")
        return df

    @staticmethod
    def ZSB_outlier_detection(verbose, dataset, threshold):
        print("ZSB_outlier_detection is performing\n")

        X = dataset.select_dtypes(['number'])
        Y = dataset.select_dtypes(['object'])

        if len(X.columns) < 1:
            print("Error: Need at least one numeric variable for ZSB detection")
            return dataset

        median = X.apply(np.median, axis=0)
        mad = 1.4296 * np.abs(X - median).apply(np.median, axis=0)
        modified_z_scores = (X - median) / mad

        outliers = X[np.abs(modified_z_scores) > 1.6]
        to_drop = outliers[(outliers.count(axis=1) / outliers.shape[1]) > threshold].index
        to_keep = set(X.index) - set(to_drop)

        if threshold == -1:
            X = X[~(np.abs(modified_z_scores) > 1.6).any(axis=1)]
        else:
            X = X.loc[list(to_keep)]

        df = X.join(Y)

        print(len(to_drop), "outlying rows have been removed")

        if len(to_drop) > 0 and verbose:
            print("with indexes:", list(to_drop))
            print("\nOutliers:\n", dataset.loc[to_drop])

        print("ZSB_outlier_detection is over\n")
        return df

    @staticmethod
    def LOF_outlier_detection(verbose, dataset, threshold):
        print("LOF_outlier_detection is performing\n")
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import RobustScaler

        cleaned_data = dataset.dropna().copy()
        if dataset.isnull().sum().sum() > 0:
            print("LOF requires no missing values, missing values have been removed using DROP.")

        X = cleaned_data.select_dtypes(include='number')
        if X.empty:
            print('Error: Need at least one numeric variable')
            return dataset

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        n_samples = len(X_scaled)
        dynamic_neighbors = max(5, min(20, int(n_samples * 0.05)))

        clf = LocalOutlierFactor(
            n_neighbors=dynamic_neighbors,
            contamination=threshold,
            novelty=False,
            metric='cosine'
        )

        labels = clf.fit_predict(X_scaled)
        outlier_mask = labels == -1
        cleaned_df = cleaned_data[~outlier_mask]

        removed_count = sum(outlier_mask)
        print(f"{removed_count} outlying rows have been removed (Threshold={threshold})")

        if verbose and removed_count > 0:
            print("Removed indexes:", cleaned_data[outlier_mask].index.tolist())
            print("\nOutliers:\n", cleaned_data[outlier_mask])

        print("LOF_outlier_detection is over\n")
        return cleaned_df

    @staticmethod
    def NaN_drop(df):
        return df.dropna()

    def _fit_lof(self, df_sample):
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import RobustScaler

        cleaned_data = df_sample.dropna().copy()
        X = cleaned_data.select_dtypes(include='number')
        if X.empty:
            return

        self.lof_scaler_ = RobustScaler()
        X_scaled = self.lof_scaler_.fit_transform(X)

        n_samples = len(X_scaled)
        dynamic_neighbors = max(5, min(20, int(n_samples * 0.05)))

        self.lof_model_ = LocalOutlierFactor(
            n_neighbors=dynamic_neighbors,
            contamination=self.threshold,
            novelty=True,
            metric='cosine'
        )
        self.lof_model_.fit(X_scaled)

    def _apply_lof_with_model(self, df_sample):
        if self.lof_model_ is None or self.lof_scaler_ is None:
            return df_sample

        df_clean = df_sample.dropna().copy()
        if df_clean.empty:
            return df_sample

        X = df_clean.select_dtypes(include='number')
        if X.empty:
            return df_sample

        X_scaled = self.lof_scaler_.transform(X)
        labels = self.lof_model_.predict(X_scaled)
        outlier_mask = labels == -1
        cleaned_df = df_clean[~outlier_mask]

        # keep rows with NaNs unchanged
        df_nan = df_sample.loc[df_sample.index.difference(df_clean.index)]
        return pd.concat([cleaned_df, df_nan], axis=0).sort_index()

    def _fit_mad(self, df_sample):
        numeric = df_sample.select_dtypes(include='number')
        if numeric.empty:
            return
        self.mad_median_ = numeric.median()
        self.mad_scale_ = (numeric - self.mad_median_).abs().median().replace(0, np.nan)
        self.mad_fill_ = numeric.mean().fillna(self.mad_median_).fillna(0.0)

    def _apply_mad_with_model(self, df_sample, n_mad=2.5):
        if self.mad_median_ is None:
            return df_sample
        out = df_sample.copy()
        numeric_cols = [col for col in self.mad_median_.index if col in out.columns]
        if not numeric_cols:
            return out
        values = out[numeric_cols]
        scale = self.mad_scale_.reindex(numeric_cols)
        median = self.mad_median_.reindex(numeric_cols)
        indicator = values.lt(median - n_mad * scale) | values.gt(median + n_mad * scale)
        for col in numeric_cols:
            if indicator[col].any():
                out.loc[indicator[col], col] = self.mad_fill_.get(col, median[col])
        return out

    def _apply_sampled_detection(self, df, method_key, sample_idx):
        sample_idx = df.index.intersection(sample_idx)
        df_sample = df.loc[sample_idx].copy()
        df_rest = df.drop(index=sample_idx).copy()

        if method_key == 'ZSB':
            df_sample = self.ZSB_outlier_detection(self.verbose, df_sample, self.threshold)
        elif method_key == 'IQR':
            df_sample = self.IQR_outlier_detection(self.verbose, df_sample, self.threshold)
        elif method_key == 'LOF':
            if self.lof_model_ is None:
                df_sample = self.LOF_outlier_detection(self.verbose, df_sample, self.threshold)
            else:
                df_sample = self._apply_lof_with_model(df_sample)
        elif method_key == 'MAD':
            df_sample = self._apply_mad_with_model(df_sample)
        else:
            raise ValueError("Invalid outlier detection method")

        return pd.concat([df_sample, df_rest], axis=0).sort_index()

    def transform(self):
        """
        apply outlier detection on dataset using 'method'(param)

        :return: df_train, df_val, df_test, using_time
        """
        df_train = self.dataset_train.copy()
        df_val = self.dataset_val.copy()
        df_test = self.dataset_test.copy()

        start_time = time.time()
        print("---carrying out outlier detection[{}]---\n".format(self.method))

        method_key = str(self.method).strip()

        sample_idx_train = df_train.sample(frac=self.e, random_state=42).index
        sample_idx_val = df_val.sample(frac=self.e, random_state=42).index
        sample_idx_test = df_test.sample(frac=self.e, random_state=42).index

        if method_key == 'LOF':
            df_train_sample = df_train.loc[sample_idx_train].copy()
            self._fit_lof(df_train_sample)
        elif method_key == 'MAD':
            df_train_sample = df_train.loc[sample_idx_train].copy()
            self._fit_mad(df_train_sample)

        df_train = self._apply_sampled_detection(df_train, method_key, sample_idx_train)
        df_val = self._apply_sampled_detection(df_val, method_key, sample_idx_val)
        df_test = self._apply_sampled_detection(df_test, method_key, sample_idx_test)

        using_time = time.time() - start_time
        print("---outlier detection [{}] is over,using time:{}---\n".format(self.method, using_time))

        return df_train, df_val, df_test, using_time


