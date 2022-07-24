import catboost
import lightgbm
import xgboost
from sklearn import dummy, ensemble, tree

from .config import N_JOBS, RANDOM_SEED, VERBOSITY

models = {
    "dc": dummy.DummyClassifier(),
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
