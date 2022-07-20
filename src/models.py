from collections import Counter

import catboost
import lightgbm
import numpy as np
import xgboost
from sklearn import ensemble, tree
from sklearn.base import BaseEstimator, ClassifierMixin

from .config import N_JOBS, RANDOM_SEED, VERBOSITY


class BaselineClassifier(BaseEstimator, ClassifierMixin):
    """Baseline model that always predicts the most common class
    with the `predict` method or training class frequencies with
    `predict_proba`.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        c = Counter(y)
        self.classes_ = list(c.keys())
        self.mode_ = c.most_common(1)[0][0]
        self.target_freq_ = np.array(list(c.values())) / X.shape[0]
        return self

    def predict(self, X):
        return self.mode_ * np.ones(X.shape[0])

    def predict_proba(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.dot(ones, self.target_freq_.reshape(1, -1))


models = {
    "bc": BaselineClassifier(),
    "dt": tree.DecisionTreeClassifier(random_state=RANDOM_SEED),
    "rf": ensemble.RandomForestClassifier(
        n_jobs=N_JOBS, random_state=RANDOM_SEED, verbose=VERBOSITY
    ),
    "xgb": xgboost.XGBClassifier(
        n_jobs=N_JOBS, random_state=RANDOM_SEED, verbosity=VERBOSITY
    ),
    "cb": catboost.CatBoostClassifier(random_state=RANDOM_SEED, verbose=VERBOSITY),
    "lgb": lightgbm.LGBMClassifier(
        n_jobs=N_JOBS, random_state=RANDOM_SEED, verbose=VERBOSITY
    ),
    "hgb": ensemble.HistGradientBoostingClassifier(
        random_state=RANDOM_SEED, verbose=VERBOSITY
    ),
}
