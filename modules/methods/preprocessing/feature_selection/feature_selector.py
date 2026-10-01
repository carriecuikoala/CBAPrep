import time
import warnings
import numpy as np
import pandas as pd


class FeatureSelector:
    """
    feature selection: LC, VAR, Tree, SVC

    parameters
    ----------
    * dataset_train (DataFrame)
        transform train data using fitted selection

    * dataset_val (DataFrame)
        transform validation data using fitted selection

    * dataset_test (DataFrame)
        transform test data using fitted selection

    * method (str, default='LC')
        strategy of feature selection:
        'LC': linear correlation filter (unsupervised)
        'VAR': variance filter (unsupervised)
        'Tree': tree-based model (supervised)
        'SVC': linear SVC (supervised)
    """

    def __init__(self, e, dataset_train, dataset_val, dataset_test, target, method='LC'):
        self.e = e
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.dataset_test = dataset_test
        self.target = target
        self.method = method
        self.threshold = 0.3
        self.verbose = False

    def get_params(self, deep=True):
        return {
            'e': self.e,
            'target': self.target,
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

    def _split_target(self, df):
        target_col = self.target
        if target_col is None or target_col not in df.columns:
            raise ValueError("Target column not found in dataset_fit.")
        y = df[target_col]
        X = df.drop(columns=[target_col])
        return X, y, target_col

    def lc(self, df_fit):
        correlation_threshold = self.threshold
        print("Apply LC feature selection with threshold=", correlation_threshold)
        numeric_df = df_fit.select_dtypes(include=[np.number])

        if numeric_df.shape[1] == 0:
            raise ValueError("No numeric columns to compute correlation.")

        corr_matrix = numeric_df.corr()

        if self.verbose:
            print("Correlation matrix")
            print(corr_matrix)

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [column for column in upper.columns if any(
            upper[column].abs() > correlation_threshold
        )]

        print('%d features with linear correlation greater than %0.2f.\n' %
              (len(to_drop), correlation_threshold))
        print('List of correlated variables to be removed :', to_drop)

        to_keep = [c for c in df_fit.columns if c not in to_drop]

        if self.verbose:
            print("List of variables to be kept")
            print(to_keep)

        return to_keep

    def var(self, df_fit):
        dn = df_fit.select_dtypes(include=[np.number])

        if dn.shape[1] == 0:
            print("No numeric columns for VAR selection. Dataset unchanged.")
            return list(df_fit.columns)

        coef = dn.std()

        print("Apply VAR feature selection with threshold=", self.threshold)

        abstract_threshold = np.percentile(coef, 100. * self.threshold)

        to_keep = coef[coef >= abstract_threshold].index.tolist()

        if self.verbose:
            print("Variables kept by VAR:", to_keep)

        return to_keep

    def tree(self, df_fit):
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.feature_selection import SelectFromModel

        print("Apply Tree-based feature selection")

        X, Y, _ = self._split_target(df_fit)
        X = X.select_dtypes(include=[np.number]).dropna()
        Y = Y.loc[X.index]

        if X.shape[1] < 1 or len(X) < 1:
            print('Error: Need at least one continous variable for feature selection')
            return []

        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(X, Y)
        model = SelectFromModel(clf, prefit=True)
        best_features = X.columns[model.get_support(indices=True)].tolist()

        if self.verbose:
            print("Best features to keep", best_features)

        return best_features

    def svc(self, df_fit):
        from sklearn.svm import LinearSVC
        from sklearn.feature_selection import SelectFromModel
        from sklearn.preprocessing import StandardScaler

        X, Y, _ = self._split_target(df_fit)
        combined = pd.concat([X, Y.rename('Target')], axis=1)
        combined_clean = combined.dropna()

        X = combined_clean.select_dtypes(include=['number']).drop('Target', axis=1, errors='ignore')
        Y = combined_clean['Target']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

        if X.empty or Y.empty:
            print("Error: No valid data after cleaning.")
            return []

        try:
            lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=10000).fit(X, Y)
            model = SelectFromModel(lsvc, prefit=True)
            best_features = X.columns[model.get_support()].tolist()
            if self.verbose:
                print("Best features to keep:", best_features)
        except Exception as e:
            print(f"SVC feature selection failed: {str(e)}")
            best_features = []

        return best_features

    def missing_ratio(self, df_fit):
        missing_ratio = df_fit.isna().mean()
        to_keep = missing_ratio[missing_ratio <= self.threshold].index.tolist()
        if not to_keep and len(df_fit.columns):
            to_keep = [missing_ratio.idxmin()]
        if self.verbose:
            print("Variables kept by missing-ratio selection:", to_keep)
        return to_keep

    def weighted_ranking(self, df_fit, k=10):
        from sklearn.feature_selection import SelectKBest, chi2

        X, y, _ = self._split_target(df_fit)
        X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        valid = X.notna().all(axis=1) & y.notna()
        X = X.loc[valid]
        y = y.loc[valid]
        if X.empty:
            return []

        # Chi-square requires non-negative values. The shift is fitted only on
        # the sampled training data and does not alter feature ranking semantics.
        X = X - X.min(axis=0)
        selector = SelectKBest(score_func=chi2, k=min(k, X.shape[1]))
        selector.fit(X, y)
        return X.columns[selector.get_support()].tolist()

    def _apply_columns(self, df, columns_to_keep, target_col=None):
        df_out = df.copy()
        if target_col is not None and target_col in df_out.columns:
            df_out = df_out.drop(columns=[target_col])
        columns = [c for c in columns_to_keep if c in df_out.columns]
        return df_out[columns]

    def transform(self):
        """
        apply feature selection on dataset using 'method'(param)

        :return: df_train, df_val, df_test, using_time
        """
        df_train = self.dataset_train.copy()
        df_val = self.dataset_val.copy()
        df_test = self.dataset_test.copy()
        df_fit = df_train.sample(frac=self.e, random_state=42).copy()

        start_time = time.time()

        print("---carrying out feature selection[{}]---\n".format(self.method))

        target_col = self.target
        method_key = str(self.method).strip()

        if method_key == 'LC':
            to_keep = self.lc(df_fit.drop(columns=[target_col], errors='ignore'))
        elif method_key == 'VAR':
            to_keep = self.var(df_fit.drop(columns=[target_col], errors='ignore'))
        elif method_key == 'Tree':
            to_keep = self.tree(df_fit)
        elif method_key == 'SVC':
            to_keep = self.svc(df_fit)
        elif method_key == 'MR':
            to_keep = self.missing_ratio(df_fit.drop(columns=[target_col], errors='ignore'))
        elif method_key == 'WR':
            to_keep = self.weighted_ranking(df_fit)
        else:
            raise ValueError('Invalid feature selection method\n')

        df_train = self._apply_columns(df_train, to_keep, target_col)
        df_val = self._apply_columns(df_val, to_keep, target_col)
        df_test = self._apply_columns(df_test, to_keep, target_col)

        using_time = time.time() - start_time
        print("---feature selection [{}] is over,using time:{}---\n".format(self.method, using_time))

        return df_train, df_val, df_test, using_time


def simple_feature_select_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep a small subset of features and encode non-numeric columns so
    feature selection can run on Google Play Store data.
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


