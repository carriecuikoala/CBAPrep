
import time
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import category_encoders as ce


class FeatureEncoder:
    """
        feature encoder:ordinal, onehot, count, target

        parameters
        ----------
        * dataset_train (dict)
            include 'train'(dataframe/list), 'test'(dataframe/list), 'target'(string/series)
            (target means target col for classification, normally equals to 'target' of dataset_train )

        * dataset_test(dict)
            include 'data'(dataframe/list), 'target'(string/series)

        * method (str, default='f1')
            strategy of feature encoder:
            'ordinal':Ordinal_Encoding(done)
            'onehot':One_Hot_Coding (done)
            'count':Count_Coding(done)
            'target':Target_Encoding(done)

    """
    def __init__(self,dataset_fit,dataset_train,dataset_val,dataset_test,target,non_num_features,method='ordinal'):
        self.dataset_fit=dataset_fit
        self.dataset_train=dataset_train
        self.dataset_val=dataset_val
        self.dataset_test=dataset_test
        self.target=target
        self.method=method
        self.non_num_features=non_num_features

    def get_params(self,deep=True):#get current params
        return{
            'method':self.method
        }

    """
    set the params for feature encoder methods
    """
    def set_params(self,**params):
        for k,v in params.items():
            if k not in self.get_params():
                warnings.warn(f"Ignore invalid params:{k}")
            else:
                setattr(self,k,v)
        return self

    """
    the functions of encoder methods
    """
    # process df_train and df_val separately
    def ordinal_encoding(self,df_fit,df_train,df_val,df_test):
        # invocate category_encoders, set category cols
        oe=ce.ordinal.OrdinalEncoder(cols=self.non_num_features)
        oe.fit(df_fit)
        df_train = oe.transform(df_train)
        df_val= oe.transform(df_val)
        df_test=oe.transform(df_test)
        return df_train, df_val,df_test

    def one_hot_coding(self,df_fit,df_train,df_val,df_test):
        # invocate category_encoders, set category cols
        ohc=ce.one_hot.OneHotEncoder(cols=self.non_num_features)
        ohc.fit(df_fit)
        df_train = ohc.transform(df_train)
        df_val = ohc.transform(df_val)
        df_test = ohc.transform(df_test)
        return df_train, df_val,df_test


    def count_coding(self,df_fit,df_train,df_val,df_test):
        cc=ce.count.CountEncoder(cols=self.non_num_features)#invocate category_encoders,set category cols
        cc.fit(df_fit)
        df_train = cc.transform(df_train)
        df_val = cc.transform(df_val)
        df_test = cc.transform(df_test)
        return df_train, df_val,df_test


    def target_encoding(self,df_fit,df_train,df_val,df_test):
        te=ce.target_encoder.TargetEncoder(cols=self.non_num_features)
        #target_Encoding is supervised coding ,needs both X and y as params
        y_fit=df_fit[self.target]
        te.fit(df_fit,y_fit)
        df_train = te.transform(df_train)
        df_val = te.transform(df_val)
        df_test = te.transform(df_test)
        return df_train, df_val,df_test

    """
    apply feature_encoder on dataset using 'method'(param)"""
    def transform(self):
        """
        parameters
        ----------
        * dataset_train (dict)
            include 'train'(dataframe/list), 'test'(dataframe/list), 'target'(string/series)
        * dataset_test(dict)
            include 'data'(dataframe/list), 'target'(string/series):target means target col

        * method (str, default='f1')
            strategy of feature encoder:
            'ordinal':Ordinal_Encoding(done)
            'onehot':One_Hot_Coding (done)
            'count':Count_Coding(done)
            'target':Target_Encoding(done)
        :return:
        """
        # copy dataset
        df_fit = self.dataset_fit.copy()
        df_train = self.dataset_train.copy()
        df_val = self.dataset_val.copy()
        df_test = self.dataset_test.copy()
        # target = self.target

        #record using time
        start_time=time.time()

        #remind the beginning of process
        print("---carrying out feature encoder[{}]---\n".format(self.method))

        # drop high-cardinality categorical features
        if self.non_num_features:
            high_card_cols = []
            n_rows = len(df_fit)
            for col in list(self.non_num_features):
                if col not in df_fit.columns:
                    continue
                nunique = df_fit[col].nunique(dropna=True)
                if (nunique > 100) or (n_rows > 0 and nunique / n_rows > 0.5):
                    high_card_cols.append(col)
            if high_card_cols:
                df_fit = df_fit.drop(columns=high_card_cols, errors='ignore')
                df_train = df_train.drop(columns=high_card_cols, errors='ignore')
                df_val = df_val.drop(columns=high_card_cols, errors='ignore')
                df_test = df_test.drop(columns=high_card_cols, errors='ignore')
                self.non_num_features = [c for c in self.non_num_features if c not in high_card_cols]

        # carry out encoding
        if self.method == 'ordinal':
            train_encoded, val_encoded,test_encoded= self.ordinal_encoding(df_fit,df_train,df_val,df_test)

        elif self.method == 'onehot':
            train_encoded, val_encoded,test_encoded = self.one_hot_coding(df_fit,df_train,df_val,df_test)

        elif self.method == 'count':
            train_encoded, val_encoded,test_encoded = self.count_coding(df_fit,df_train, df_val,df_test)

        elif self.method == 'target':
            train_encoded, val_encoded,test_encoded = self.target_encoding(df_fit,df_train, df_val,df_test)

        else :
            raise ValueError('Invalid feature coding method\n')

        # align columns across splits after encoding
        all_columns = train_encoded.columns.union(val_encoded.columns).union(test_encoded.columns)
        train_encoded = train_encoded.reindex(columns=all_columns, fill_value=0)
        val_encoded = val_encoded.reindex(columns=all_columns, fill_value=0)
        test_encoded = test_encoded.reindex(columns=all_columns, fill_value=0)
        print("---feature encoder [{}] is over,using time:{}---\n".format(self.method,time.time()-start_time))

        return train_encoded,val_encoded,test_encoded,time.time()-start_time

