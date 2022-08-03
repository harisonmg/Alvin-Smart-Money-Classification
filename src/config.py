from logging.config import dictConfig
from pathlib import Path

import decouple
from sklearn import model_selection

# file paths
BASE_DIR = Path(__file__).parent.parent

INPUT_DIR = BASE_DIR / "input"

OUTPUT_DIR = BASE_DIR / "output"

# data
TRAIN_DATA = INPUT_DIR / "train.csv"

TEST_DATA = INPUT_DIR / "test.csv"

# columns in the data
INDEX_COL = "Transaction_ID"

TARGET_COL = "MERCHANT_CATEGORIZED_AS"

DATETIME_COLS = ["MERCHANT_CATEGORIZED_AT", "PURCHASED_AT"]

# project details
PROJECT_NAME = "alvin-smcc"

# random seed
RANDOM_SEED = 98765

# cross validation
NUM_FOLDS = decouple.config("NUM_FOLDS", cast=int, default=3)

CV_SPLITTER = model_selection.StratifiedKFold(
    n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED
)

# metrics
EVAL_METRICS = ("neg_log_loss",)

# parallel jobs
N_JOBS = decouple.config("N_JOBS", cast=int, default=-1)

# logging
LOG_DIR = decouple.config("LOG_DIR", default=OUTPUT_DIR / "logs")

if not LOG_DIR.exists():
    LOG_DIR.mkdir(parents=True)  # create log directory if it doesn't exist

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "console": {"format": "%(levelname)s: %(message)s"},
        "file": {"format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "console"},
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "file",
            "filename": LOG_DIR / "file.log",
            "maxBytes": 1024**2,  # 1 MB
            "backupCount": 10,
        },
    },
    "loggers": {
        "src": {
            "handlers": [
                "console",
                "file",
            ],
            "level": "INFO",
        },
        "optuna": {
            "handlers": ["file"],
            "level": "INFO",
        },
    },
}

dictConfig(LOGGING)

VERBOSITY = decouple.config("VERBOSITY", cast=int, default=1)

VERBOSE = VERBOSITY > 1

# MLflow
MLFLOW_TRACKING_URI = decouple.config(
    "MLFLOW_TRACKING_URI", default=f"sqlite:///{OUTPUT_DIR}/mlruns.db"
)

# optuna
OPTUNA_DATABASE_URL = decouple.config(
    "OPTUNA_DATABASE_URL", default=f"sqlite:///{OUTPUT_DIR}/optuna.db"
)
