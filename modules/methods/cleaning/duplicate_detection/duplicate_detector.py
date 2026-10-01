import re
import time

import numpy as np
import pandas as pd


class DuplicateDetector:
    """Apply sampled exact or approximate duplicate removal to each split."""

    def __init__(self, e, dataset_train, dataset_val, dataset_test, method="ED"):
        self.e = float(e)
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.method = method
        self.similarity_threshold = 0.90
        self.string_prefix_length = 16
        self.numeric_tolerance_ratio = 0.001

    def get_params(self, deep=True):
        return {
            "e": self.e,
            "method": self.method,
            "similarity_threshold": self.similarity_threshold,
            "string_prefix_length": self.string_prefix_length,
            "numeric_tolerance_ratio": self.numeric_tolerance_ratio,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key in self.get_params():
                setattr(self, key, value)
        return self

    def _approximate_drop(self, sample):
        if sample.empty:
            return sample
        canonical = pd.DataFrame(index=sample.index)
        for column in sample.columns:
            series = sample[column]
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce")
                lower = numeric.min()
                span = numeric.max() - lower
                tolerance = max(float(span) * self.numeric_tolerance_ratio, 1e-12)
                canonical[column] = np.rint((numeric - lower) / tolerance).fillna(-1)
            else:
                text = (
                    series.astype("string")
                    .fillna("")
                    .str.lower()
                    .str.replace(r"[^0-9a-z]+", "", regex=True)
                )
                canonical[column] = (
                    text.str.slice(0, self.string_prefix_length)
                    + ":"
                    + (text.str.len() // 4).astype(str)
                )
        keep_mask = ~canonical.duplicated(keep="first")
        return sample.loc[keep_mask]

    def _apply(self, df, method):
        if df.empty or self.e <= 0:
            return df.copy()
        sample_idx = df.sample(frac=min(self.e, 1.0), random_state=42).index
        sample = df.loc[sample_idx].copy()
        rest = df.drop(index=sample_idx).copy()
        if method == "ED":
            sample = sample.drop_duplicates(keep="first")
        elif method == "AD":
            sample = self._approximate_drop(sample)
        else:
            raise ValueError("Invalid duplicate detection method")
        return pd.concat([sample, rest], axis=0).sort_index()

    def transform(self):
        started = time.time()
        method = str(self.method).strip().upper()
        train = self._apply(self.dataset_train, method)
        val = self._apply(self.dataset_val, method)
        test = self._apply(self.dataset_test, method)
        return train, val, test, time.time() - started
