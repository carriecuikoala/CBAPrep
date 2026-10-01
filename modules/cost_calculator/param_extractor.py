from modules.methods.preprocessing.feature_coding.feature_encoder import FeatureEncoder
from modules.methods.preprocessing.data_normalization.data_normalizer import DataNormalization
from modules.methods.preprocessing.feature_selection.feature_selector import FeatureSelector
from modules.methods.preprocessing.feature_extraction.feature_extractor import FeatureExtractor
from modules.methods.cleaning.nan_handling.imputer import Imputer
from modules.methods.cleaning.outlier_detection.outlier_detector import OutlierDetector
from modules.methods.cleaning.duplicate_detection.duplicate_detector import DuplicateDetector
from modules.methods.preprocessing.data_conversion.data_converter import DataConverter
from modules.methods.preprocessing.dataset_division.dataset_divider import DatasetDivider

import pandas as pd
import numpy as np
import pickle
import os
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle as pkl
from modules.common.dataset_utils import resolve_dataset_name


def get_pkl_store_dir(project_root):
    pkl_dir = os.path.join(project_root, "artifacts", "pkl_store")
    os.makedirs(pkl_dir, exist_ok=True)
    return pkl_dir


def get_versioned_cost_model_dir(project_root, artifact_version):
    version = str(artifact_version or "").strip()
    if not version or not version.replace("_", "").replace("-", "").isalnum():
        raise ValueError("artifact_version may contain only letters, numbers, '_' and '-'.")
    model_dir = os.path.join(project_root, "artifacts", "cost_models", version, "profiles")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


class ParamExtractor:
    """
        param extractor
        ----------
        extract cost params of auto methods for machine_cost.py

        parameters
        ----------
        * dataset_train (dict)
            include 'train'(dataframe/list), 'test'(dataframe/list), 'target'(string/series)
            (target means target col for classification, normally equals to 'target' of dataset_train )

        * dataset_test(dict)
            include 'data'(dataframe/list), 'target'(string/series)

        * process (str, default='f1')
            exact process to extract:
            'feature coding':feature_encoder
            'data normalization':data_normalizer
            'feature selection':feature_selector
            'feature extraction':feature_extractor
            'imputation':imputer
            'outlier detection':outlier_detector
            'data conversion':data_converter
            'dataset division':dataset_divider
        """
    def __init__(self,dataset_train,dataset_val,dataset_test,target,process):
        self.dataset_train=dataset_train
        self.dataset_val=dataset_val
        self.dataset_test=dataset_test
        self.target=target
        self.process=process

    def get_params(self, deep=True):
        return {
            'dataset_train':self.dataset_train,
            'dataset_val': self.dataset_val,
            'dataset_test':self.dataset_test,
            'process': self.process}

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn(f"Ignore invalid params:{k}")
            else:
                setattr(self, k, v)
        return self

    def method_run(self, method, step,**kwargs):#
        """
        process sub_dataset on ratios,
        marge it with other unprocessed part into final data
        return final train_data and test_data

        **kwargs:
        include random_state,non_num_features,run_num
        """
        # restore process time
        process_time = []
        repeat_time_samples = []
        e_values = []

        # get output_dir,random_state,non_num_features,run_num from [**kwargs] if provided
        output_dir = kwargs.get('output_dir', '../../data/partial_processed')
        random_state = kwargs.get('random_state', 42)


        run_num=kwargs.get('run_num',10)
        cost_param = str(kwargs.get('cost_param', 'e')).lower()
        artifact_version = kwargs.get('artifact_version')
        save_repeat_details = bool(kwargs.get('save_repeat_details', artifact_version is not None))

        # create folder for output
        os.makedirs(output_dir, exist_ok=True)

        # generate suffix for output file
        suffix = f"{method}_step{int(step * 100)}"

        #do the process
        if self.process == 'data normalization':

            non_num_features = kwargs.get('non_num_features')

            #run once every e between 0 and 1
            for e in np.arange(step, 1 + step, step):

                # sample dataset with ratios
                sampled_data_train = self.dataset_train.sample(frac=e, random_state=random_state).copy()
                sampled_data_val = self.dataset_val.sample(frac=e, random_state=random_state).copy()
                sampled_data_test = self.dataset_test.sample(frac=e, random_state=random_state).copy()

                dn = DataNormalization(sampled_data_train, self.dataset_train, self.dataset_val, self.dataset_test)
                runtimes = []

                # run the method [run_num] times,take average time
                for i in range(run_num):#  record process time

                    dn.set_params(method=method)
                    _, _,_,runtime=dn.transform()
                    runtimes.append(float(runtime))
                    # print("method:{} e:{} times:{} runtime:{}".format(method,e,i,runtime))

                final_time=float(np.mean(runtimes))
                process_time.append(final_time)
                repeat_time_samples.append(runtimes)
                e_values.append(float(e))

        elif self.process=='feature coding':

            non_num_features = kwargs.get('non_num_features')

            for e in np.arange(step, 1 + step, step):


                sampled_data_train = self.dataset_train.sample(frac=e, random_state=random_state).copy()# sample dataset with ratios
                fe = FeatureEncoder(sampled_data_train, self.dataset_train, self.dataset_val, self.dataset_test, self.target, non_num_features)
                runtimes = []

                for i in range(run_num):  # run [run_num] times,take average time

                    fe.set_params(method=method)
                    _, _, _, runtime = fe.transform()# record process time
                    runtimes.append(float(runtime))
                    # print("method:{} e:{} times:{} runtime:{}".format(method,e,i,runtime))

                final_time = float(np.mean(runtimes))
                process_time.append(final_time)
                repeat_time_samples.append(runtimes)
                e_values.append(float(e))

        elif self.process=='feature selection':

            for e in np.arange(step, 1 + step, step):

                fs = FeatureSelector(e, self.dataset_train, self.dataset_val, self.dataset_test, self.target)
                runtimes = []

                for i in range(run_num):
                    fs.set_params(method=method)
                    _, _, _, runtime = fs.transform()
                    runtimes.append(float(runtime))

                final_time = float(np.mean(runtimes))
                process_time.append(final_time)
                repeat_time_samples.append(runtimes)
                e_values.append(float(e))

        elif self.process=='feature extraction':

            for e in np.arange(step, 1 + step, step):

                fe = FeatureExtractor(e, self.dataset_train, self.dataset_val, self.dataset_test, self.target)
                runtimes = []

                for i in range(run_num):
                    fe.set_params(method=method)
                    _, _, _, runtime = fe.transform()
                    runtimes.append(float(runtime))

                final_time = float(np.mean(runtimes))
                process_time.append(final_time)
                repeat_time_samples.append(runtimes)
                e_values.append(float(e))

        elif self.process=='imputation':

            for e in np.arange(step, 1 + step, step):

                imp = Imputer(e, self.dataset_train, self.dataset_val, self.dataset_test)
                runtimes = []

                for i in range(run_num):
                    imp.set_params(method=method)
                    _, _, _, runtime = imp.transform()
                    runtimes.append(float(runtime))

                final_time = float(np.mean(runtimes))
                process_time.append(final_time)
                repeat_time_samples.append(runtimes)
                e_values.append(float(e))

        elif self.process=='outlier detection':

            for e in np.arange(step, 1 + step, step):

                od = OutlierDetector(e, self.dataset_train, self.dataset_val, self.dataset_test)
                runtimes = []

                for i in range(run_num):
                    od.set_params(method=method)
                    _, _, _, runtime = od.transform()
                    runtimes.append(float(runtime))

                final_time = float(np.mean(runtimes))
                process_time.append(final_time)
                repeat_time_samples.append(runtimes)
                e_values.append(float(e))

        elif self.process == 'duplicate detection':

            for e in np.arange(step, 1 + step, step):
                detector = DuplicateDetector(
                    e, self.dataset_train, self.dataset_val, self.dataset_test, method=method
                )
                runtimes = []
                for _ in range(run_num):
                    _, _, _, runtime = detector.transform()
                    runtimes.append(float(runtime))
                process_time.append(float(np.mean(runtimes)))
                repeat_time_samples.append(runtimes)
                e_values.append(float(e))

        elif self.process=='data conversion':

            suffix = f"{method}_full"
            dc = DataConverter(self.dataset_train)
            runtimes = []

            for i in range(run_num):
                _, runtime = dc.transform()
                runtimes.append(float(runtime))

            final_time = float(np.mean(runtimes))
            process_time.append(final_time)
            repeat_time_samples.append(runtimes)
            e_values.append(1.0)

        elif self.process=='dataset division':

            suffix = f"{method}_full"
            test_rate = kwargs.get('test_rate', 0.3)
            val_rate = kwargs.get('val_rate', 0.3)
            dd = DatasetDivider(self.dataset_train, test_rate=test_rate, val_rate=val_rate)
            runtimes = []

            for i in range(run_num):
                _, _, _, runtime = dd.transform()
                runtimes.append(float(runtime))

            final_time = float(np.mean(runtimes))
            process_time.append(final_time)
            repeat_time_samples.append(runtimes)
            e_values.append(1.0)

        else:
            raise ValueError

        dataset_name = resolve_dataset_name(kwargs.get('dataset_name', None))
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        if artifact_version:
            pkl_store_dir = get_versioned_cost_model_dir(project_root, artifact_version)
        else:
            pkl_store_dir = get_pkl_store_dir(project_root)
        if dataset_name:
            output_name = f"time_{dataset_name}_{suffix}_{cost_param}.pkl"
        else:
            output_name = f"time_{suffix}_{cost_param}.pkl"
        with open(os.path.join(pkl_store_dir, output_name), "wb") as f:
            pkl.dump(process_time, f)# save recorded time
        if save_repeat_details:
            profile_name = output_name[:-4] + ".profile.pkl"
            profile = {
                "schema_version": 1,
                "artifact_version": artifact_version or "legacy",
                "dataset": dataset_name,
                "process": self.process,
                "method": method,
                "cost_param": cost_param,
                "step": float(step),
                "run_num": int(run_num),
                "e_values": e_values,
                "mean_times": process_time,
                "repeat_times": repeat_time_samples,
            }
            with open(os.path.join(pkl_store_dir, profile_name), "wb") as f:
                pkl.dump(profile, f)
        return process_time

        # with open(f"time_{suffix}.pkl", "wb") as f:
        #     pkl.dump(process_time, f)


    def Visualization(self,method,process_time):#dataset size as x axis，running time as y axis

        # knn_times = [
        #     0.011601396998157725, 0.035249064000090584, 0.08704489399911836,
        #     0.13969299099873753, 0.21208820299827494, 0.3055772009992506,
        #     0.408122497999575, 0.5220553709985688, 0.4665423110022675
        # ]
        l=len(process_time)
        start=100/l
        end=100+start
        dataset_sizes = np.arange(start, end, start)


        plt.figure(figsize=(20, 12))
        plt.plot(dataset_sizes, process_time,
                 marker='o',
                 color='#2c7bb6',
                 linestyle='--',
                 linewidth=2,
                 markersize=8,
                 markerfacecolor='#d7191c')


        plt.xlabel("Dataset Size", fontsize=12, fontweight='bold')
        plt.ylabel("Running Time (seconds)", fontsize=12, fontweight='bold')
        plt.title(f'{method} Running Time vs. Dataset Size', fontsize=14, pad=20)


        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(dataset_sizes)
        plt.tight_layout()

        # plt.savefig("knn_runtime_analysis.png", dpi=300, bbox_inches='tight')

        plt.show()

    def show_param(self,step,method,data):


        x = np.linspace(step, 1, len(data))
        y = np.array(data)

        # np.polyfit
        slope, intercept = np.polyfit(x, y, 1)

        y_fit = slope * x + intercept

        print(f"fitting equation: y = {slope:.6f}x + ({intercept:.6f})")
        print(f"slope (k): {slope}")
        print(f"intercept (b): {intercept}")

        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, color='red', label='Test Data', zorder=5)  # init point
        plt.plot(x, y_fit, color='blue', linestyle='--', label='Fitted Line')  # fitting line
        plt.xticks(np.linspace(step, 1, 11))

        method_bold = method.replace('_', r'\_')
        plt.title(f"{self.process} method " + r"$\bf{" + method_bold + r"}$: e vs time", fontsize=14)
        plt.xlabel('e (x)', fontsize=12)
        plt.ylabel('time (y)', fontsize=12)
        plt.text(step, max(y) * 0.9, f'$y = {slope:.4f}x + ({intercept:.4f})$', fontsize=12, color='blue')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.show()
class DatasetPre:
    def __init__(self,dataset):
        self.df=dataset

    def preprocess_google_play_data(self):

        df=self.df
        df['Android Ver'] = df['Android Ver'].fillna('Unknown')

        df = df[~df['Installs'].isin(['Paid', 'Free'])]
        df['Installs'] = df['Installs'].str.replace('[+,]', '', regex=True).astype(int)

        df['Price'] = df['Price'].replace('[\$,]', '', regex=True).astype(float)

        df['Size'] = (
            df['Size']
            .replace('Varies with device', np.nan)
            .str.replace('k', '')
            .str.replace('M', '')
            .astype(float)
            .apply(lambda x: x / 1024 if x < 100 else x)  # 处理k单位的转换
        )
        return  df

    def simple_feature_select_and_encode(self) -> pd.DataFrame:
        """
        Keep a small subset of features and encode non-numeric columns so
        normalization can run on Google Play Store data.
        """
        df=self.df
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



