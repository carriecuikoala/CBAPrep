
import time
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os

class DataNormalization:
    """
    data normalization: Z-score, Min-Max, Robust, MaxAbs

    parameters
    ----------
    * dataset (dict)
        include 'train'(dataframe), 'test'(dataframe), 'target'(series)

    * method (str, default='f1')
        strategy of data normalization:
        'z_score':Z_score (done)
        'robust':Robust (done)
        'min_max':Min_Max (done)
        'max_abs':MaxAbs (done)
        'normalizer':Normalizer (done)
        'sigmoid':sigmoid (done)

    * verbose(bool,default='True')
        whether print detailed information(using time)
    """
    def __init__(self, dataset_fit,dataset_train,dataset_val,dataset_test, method='f1'):
        self.dataset_fit = dataset_fit
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.method=method

    def get_params(self, deep=True):# get current params
        return{
            'method':self.method,
        }

    """
    set the params for normalization methods
    """
    def set_params(self, **params):
        for k,v in params.items():
            if k not in self.get_params():
                warnings.warn(f"Ignore invalid params:{k}")
            else:
                setattr(self,k,v)
        return self

    """
    the functions of normalization methods
    """
    def z_score(self,df_fit,df_train,df_val,df_test):
        scaler=preprocessing.StandardScaler()
        scaler.fit(df_fit)
        df_train=scaler.transform(df_train)
        df_val=scaler.transform(df_val)
        df_test = scaler.transform(df_test)
        return df_train,df_val,df_test

    def robust(self,df_fit,df_train,df_val,df_test):
        scaler=preprocessing.RobustScaler()
        scaler.fit(df_fit)
        df_train=scaler.transform(df_train)
        df_val=scaler.transform(df_val)
        df_test = scaler.transform(df_test)
        return df_train,df_val,df_test

    def min_max(self,df_fit,df_train,df_val,df_test):
        scaler=preprocessing.MinMaxScaler()
        scaler.fit(df_fit)
        df_train=scaler.transform(df_train)
        df_val=scaler.transform(df_val)
        df_test = scaler.transform(df_test)
        return df_train,df_val,df_test

    def max_abs(self,df_fit,df_train,df_val,df_test):
        scaler=preprocessing.MaxAbsScaler()
        scaler.fit(df_fit)
        df_train=scaler.transform(df_train)
        df_val=scaler.transform(df_val)
        df_test = scaler.transform(df_test)
        return df_train,df_val,df_test

    def normalizer(self,df_fit,df_train,df_val,df_test):
        scaler=preprocessing.Normalizer()
        scaler.fit(df_fit)
        df_train=scaler.transform(df_train)
        df_val=scaler.transform(df_val)
        df_test = scaler.transform(df_test)
        return df_train,df_val,df_test

    @staticmethod
    def _apply_transformer(transformer, df_fit, df_train, df_val, df_test):
        transformer.fit(df_fit)
        return (
            transformer.transform(df_train),
            transformer.transform(df_val),
            transformer.transform(df_test),
        )

    def quantile(self, df_fit, df_train, df_val, df_test, output_distribution):
        n_quantiles = max(1, min(1000, len(df_fit)))
        transformer = preprocessing.QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            random_state=0,
        )
        return self._apply_transformer(transformer, df_fit, df_train, df_val, df_test)

    def power(self, df_fit, df_train, df_val, df_test):
        transformer = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True)
        return self._apply_transformer(transformer, df_fit, df_train, df_val, df_test)

    def kbins(self, df_fit, df_train, df_val, df_test, strategy):
        transformer = preprocessing.KBinsDiscretizer(
            n_bins=5,
            encode='ordinal',
            strategy=strategy,
            subsample=None,
        )
        return self._apply_transformer(transformer, df_fit, df_train, df_val, df_test)

    def sigmoid(self,df_fit,df_train,df_val,df_test):
        tensor_fit = torch.tensor(df_fit.to_numpy(dtype=np.float32), dtype=torch.float32)
        mean = torch.mean(tensor_fit)
        std = torch.std(tensor_fit)
        if std.item() == 0:
            std = torch.tensor(1.0, dtype=torch.float32)

        def transform(df):
            # 转换为张量
            tensor_data = torch.tensor(df.to_numpy(dtype=np.float32), dtype=torch.float32)

            # 2. Z-Score 标准化: (x - mean) / std
            z_scored = (tensor_data - mean) / std

            # 3. Sigmoid 映射: 1 / (1 + exp(-x))
            normalized = torch.sigmoid(z_scored)

            # 转回 DataFrame 保持格式一致
            return pd.DataFrame(normalized.numpy(), columns=df.columns, index=df.index)

        # 对各数据集应用变换
        df_train = transform(df_train)
        df_val = transform(df_val)
        df_test = transform(df_test)

        return df_train,df_val,df_test

    """
    apply normalization on dataset using 'method'(param)
    """
    def transform(self):

        """
         * method (str, default='f1')
        strategy of data normalization:
        'z_score':z_score
        'robust':robust
        'min_max':min_max
        'max_abs':max_abs
        'normalizer':normalizer
        'sigmoid':sigmoid

        :return: dataset after process
        """

        # copy dataset for normalization
        df_fit=self.dataset_fit.copy()
        df_train = self.dataset_train.copy()
        df_val = self.dataset_val.copy()
        df_test = self.dataset_test.copy()

        # record using time
        start_time=time.time()

        # remind the beginning of process
        print("---carrying out data normalization[{}]---\n".format(self.method))

        if self.method =='z_score':
            train_normalized,val_normalized,test_normalized=self.z_score(df_fit,df_train,df_val,df_test)

        elif self.method =='robust':
            train_normalized,val_normalized,test_normalized=self.robust(df_fit,df_train,df_val,df_test)

        elif self.method =='min_max':
            train_normalized,val_normalized,test_normalized=self.min_max(df_fit,df_train,df_val,df_test)

        elif self.method =='max_abs':
            train_normalized,val_normalized,test_normalized=self.max_abs(df_fit,df_train,df_val,df_test)

        elif self.method =='normalizer':
            train_normalized,val_normalized,test_normalized=self.normalizer(df_fit,df_train,df_val,df_test)

        elif self.method =='sigmoid':
            train_normalized,val_normalized,test_normalized=self.sigmoid(df_fit,df_train,df_val,df_test)

        elif self.method == 'quantile_uniform':
            train_normalized, val_normalized, test_normalized = self.quantile(
                df_fit, df_train, df_val, df_test, output_distribution='uniform'
            )

        elif self.method == 'quantile_normal':
            train_normalized, val_normalized, test_normalized = self.quantile(
                df_fit, df_train, df_val, df_test, output_distribution='normal'
            )

        elif self.method == 'power':
            train_normalized, val_normalized, test_normalized = self.power(
                df_fit, df_train, df_val, df_test
            )

        elif self.method == 'kbins_uniform':
            train_normalized, val_normalized, test_normalized = self.kbins(
                df_fit, df_train, df_val, df_test, strategy='uniform'
            )

        elif self.method == 'kbins_quantile':
            train_normalized, val_normalized, test_normalized = self.kbins(
                df_fit, df_train, df_val, df_test, strategy='quantile'
            )

        else:
            raise ValueError('Invalid normalization method\n')

        print("---data normalization [{}] is over,using time:{}---\n".format(self.method, time.time() - start_time))

        return train_normalized,val_normalized,test_normalized,time.time() - start_time


def simple_feature_select_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep a small subset of features and encode non-numeric columns so
    normalization can run on Google Play Store data.
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


