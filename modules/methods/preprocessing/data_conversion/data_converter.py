import time
import warnings
import numpy as np
import pandas as pd


class DataConverter:
    """
    data conversion for google play store dataset

    parameters
    ----------
    * dataset (DataFrame)
        input dataset to be converted
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        for k, v in params.items():
            warnings.warn(f"Ignore invalid params:{k}")
        return self

    def preprocess_google_play_data(self, df):
        df = df.copy()
        # If already converted dataset (e.g., data.csv with Install numeric), skip conversion.
        if 'Installs' not in df.columns and 'Install' in df.columns:
            return df
        if 'Android Ver' in df.columns:
            df['Android Ver'] = df['Android Ver'].fillna('Unknown')

        if 'Installs' in df.columns:
            df = df[~df['Installs'].isin(['Paid', 'Free'])]
            df['Installs'] = df['Installs'].astype(str).str.replace('[+,]', '', regex=True)
            df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce').fillna(0).astype(int)

        if 'Price' in df.columns:
            df['Price'] = df['Price'].replace(r'[\$,]', '', regex=True).astype(float)

        if 'Size' in df.columns:
            df['Size'] = (
                df['Size']
                .replace('Varies with device', np.nan)
                .astype(str)
                .str.replace('k', '')
                .str.replace('M', '')
                .replace('nan', np.nan)
                .astype(float)
                .apply(lambda x: x / 1024 if x < 100 else x)
            )
        return df

    def transform(self):
        """
        apply data conversion

        :return: df, using_time
        """
        df = self.dataset.copy()

        start_time = time.time()
        print("---carrying out data conversion---\n")

        df = self.preprocess_google_play_data(df)

        using_time = time.time() - start_time
        print("---data conversion is over,using time:{}---\n".format(using_time))

        return df, using_time


