import catboost
import lightgbm
import xgboost
from sklearn import ensemble, linear_model, tree

from .config import N_JOBS, RANDOM_SEED, VERBOSITY

models = {
    "lr": linear_model.LogisticRegression(random_state=RANDOM_SEED),
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
