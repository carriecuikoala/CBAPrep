import time
import warnings
from sklearn.model_selection import train_test_split
import pandas as pd


class DatasetDivider:
    """
    dataset division

    parameters
    ----------
    * dataset (DataFrame)
        input dataset to split
    * test_rate (float)
        ratio for test set
    * val_rate (float)
        ratio for validation set (w.r.t full dataset)
    """

    def __init__(self, dataset, test_rate=0.3, val_rate=0.3):
        self.dataset = dataset
        self.test_rate = test_rate
        self.val_rate = val_rate

    def get_params(self, deep=True):
        return {
            'test_rate': self.test_rate,
            'val_rate': self.val_rate,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn(f"Ignore invalid params:{k}")
            else:
                setattr(self, k, v)
        return self

    def transform(self):
        """
        apply dataset division

        :return: df_train, df_val, df_test, using_time
        """
        df = self.dataset.copy()

        start_time = time.time()
        print("---carrying out dataset division---\n")

        if not (0 < self.test_rate < 1):
            raise ValueError("test_rate must be in (0, 1)")
        if not (0 <= self.val_rate < 1):
            raise ValueError("val_rate must be in [0, 1)")
        if self.test_rate + self.val_rate >= 1:
            raise ValueError("test_rate + val_rate must be < 1")

        data_train, df_test = train_test_split(
            df, test_size=self.test_rate, random_state=42
        )
        if self.val_rate == 0:
            df_train = data_train
            df_val = df.iloc[0:0].copy()
        else:
            val_ratio = self.val_rate / (1 - self.test_rate)
            df_train, df_val = train_test_split(
                data_train, test_size=val_ratio, random_state=42
            )

        using_time = time.time() - start_time
        print("---dataset division is over,using time:{}---\n".format(using_time))

        return df_train, df_val, df_test, using_time


