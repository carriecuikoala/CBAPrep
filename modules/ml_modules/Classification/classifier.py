#!/usr/bin/env python3
# coding: utf-8

import time
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score


SUPPORTED_STRATEGIES = ("LDA", "CART", "NB", "MNB", "LR", "RF", "XGB", "MLP", "SVM")

STRATEGY_ALIASES = {
    "RANDOM_FOREST": "RF",
    "RANDOMFOREST": "RF",
    "XGBOOST": "XGB",
    "XGBCLASSIFIER": "XGB",
    "SVC": "SVM",
}


def normalize_strategy_name(strategy):
    name = str(strategy or "NB").strip().upper().replace("-", "_").replace(" ", "_")
    return STRATEGY_ALIASES.get(name, name)


class Classifier:
    """
    Classification task using a particular method.
    """

    def __init__(self, dataset, target, strategy='NB', k_folds=10, verbose=False):
        self.dataset = dataset
        self.target = target
        self.strategy = normalize_strategy_name(strategy)
        self.k_folds = k_folds
        self.verbose = verbose
        self._last_y_true = None
        self._last_y_pred = None

    def get_params(self, deep=True):
        return {
            'strategy': self.strategy,
            'target': self.target,
            'k_folds': self.k_folds,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn(
                    "Invalid parameter(s) for classifier. IGNORED. "
                    "Check classifier.get_params().keys()"
                )
            else:
                setattr(self, k, v)

    def _prepare_data(self, dataset):
        X_train = dataset['train'].select_dtypes(['number']).dropna()
        if len(X_train.columns) == 0:
            return None, None, None, None
        y_train = dataset['target'].loc[X_train.index]

        X_test = dataset['test'].select_dtypes(['number']).dropna()
        if isinstance(dataset.get('target_test'), dict):
            y_test = dataset['target'].loc[X_test.index]
        else:
            y_test = dataset.get('target_test')
            if hasattr(y_test, "loc"):
                y_test = y_test.loc[X_test.index]

        if self.target in X_train.columns:
            X_train = X_train.drop(columns=[self.target])
        if self.target in X_test.columns:
            X_test = X_test.drop(columns=[self.target])

        return X_train, y_train, X_test, y_test

    def _fit_and_score(self, model, dataset, encode_target=False):
        X_train, y_train, X_test, y_test = self._prepare_data(dataset)
        if X_train is None or len(X_train) < 2 or X_train.shape[1] == 0:
            return None

        self._last_y_true = None
        self._last_y_pred = None

        if X_test is not None and len(X_test) > 0:
            if encode_target:
                encoder = LabelEncoder()
                y_train_fit = encoder.fit_transform(y_train)
                model.fit(X_train, y_train_fit)
                y_pred_fit = np.asarray(model.predict(X_test), dtype=int)
                y_pred = encoder.inverse_transform(y_pred_fit)
                # Test-only classes cannot be predicted, so they correctly count
                # as errors instead of failing label transformation.
                score = accuracy_score(y_test, y_pred)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
            self._last_y_true = y_test
            self._last_y_pred = y_pred
            return float(score)

        # fallback to CV if no test data
        k = min(self.k_folds, len(np.unique(y_train)))
        if k < 2:
            return None
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
        scores = cross_val_score(model, X_train, y_train, cv=cv)
        return float(np.mean(scores))

    def LDA_classification(self, dataset, target):
        model = LinearDiscriminantAnalysis()
        return self._fit_and_score(model, dataset)

    def CART_classification(self, dataset, target):
        model = DecisionTreeClassifier(random_state=1)
        return self._fit_and_score(model, dataset)

    def NB_classification(self, dataset, target):
        model = GaussianNB()
        return self._fit_and_score(model, dataset)

    def MNB_classification(self, dataset, target):
        model = MultinomialNB()
        return self._fit_and_score(model, dataset)

    def LR_classification(self, dataset, target):
        model = LogisticRegression(max_iter=200, solver='lbfgs')
        return self._fit_and_score(model, dataset)

    def RF_classification(self, dataset, target):
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=1,
            n_jobs=-1,
        )
        return self._fit_and_score(model, dataset)

    def XGB_classification(self, dataset, target):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGB requires the optional 'xgboost' package. "
                "Install it in the cbas environment before running XGB experiments."
            ) from exc

        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=1,
            n_jobs=-1,
        )
        return self._fit_and_score(model, dataset, encode_target=True)

    def MLP_classification(self, dataset, target):
        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=300,
            random_state=1,
        )
        return self._fit_and_score(model, dataset)

    def SVM_classification(self, dataset, target):
        model = SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=1024)
        return self._fit_and_score(model, dataset)

    def transform(self, return_metrics=False):
        start_time = time.time()
        d = self.dataset
        if self.verbose:
            print("\n==== Data Diagnostics ====")
            print("Train shape: " + str(d['train'].shape))
            print("Numeric columns: " + str(d['train'].select_dtypes(['number']).columns.tolist()))
            print("Missing values:\n" + str(d['train'].isnull().sum()))

        if self.target != d['target'].name:
            raise ValueError("Target variable invalid.")

        if self.strategy == "LDA":
            acc = self.LDA_classification(dataset=d, target=self.target)
        elif self.strategy == "CART":
            acc = self.CART_classification(dataset=d, target=self.target)
        elif self.strategy == "NB":
            acc = self.NB_classification(dataset=d, target=self.target)
        elif self.strategy == "MNB":
            acc = self.MNB_classification(dataset=d, target=self.target)
        elif self.strategy == "LR":
            acc = self.LR_classification(dataset=d, target=self.target)
        elif self.strategy == "RF":
            acc = self.RF_classification(dataset=d, target=self.target)
        elif self.strategy == "XGB":
            acc = self.XGB_classification(dataset=d, target=self.target)
        elif self.strategy == "MLP":
            acc = self.MLP_classification(dataset=d, target=self.target)
        elif self.strategy == "SVM":
            acc = self.SVM_classification(dataset=d, target=self.target)
        else:
            expected = ", ".join(SUPPORTED_STRATEGIES)
            raise ValueError(f"The classification function should be one of: {expected}.")

        if self.verbose:
            print("Classification done -- CPU time: %s seconds" % (time.time() - start_time))

        if return_metrics:
            if self._last_y_true is not None and self._last_y_pred is not None:
                f1 = f1_score(self._last_y_true, self._last_y_pred, average='macro')
            else:
                f1 = None
            return {"accuracy": acc, "f1": f1}
        return acc
