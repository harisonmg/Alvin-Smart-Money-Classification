import functools
import logging
import time

import mlflow

from . import config

# logger
logger = logging.getLogger(__name__)


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info(f"Finished {func.__name__!r} in {run_time:.4f} seconds")
        return value

    return wrapper


def configure_mlflow(
    func, tracking_uri=config.MLFLOW_TRACKING_URI, experiment_name=config.PROJECT_NAME
):
    """Configure MLflow before running a function"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # set the tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI has been set to: {tracking_uri!r}")

        # set the experiment name
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment has been set to: {experiment_name!r}")
        return func(*args, **kwargs)

    return wrapper


def load_model(run_id, model_name="model"):
    """Load a scikit-learn model from a given run ID and model name"""
    return mlflow.sklearn.load_model(f"runs:/{run_id}/{model_name}")


def load_models(run_id: str) -> list:
    """Load all models from a given run ID"""
    # get the number of folds for this run
    n_folds = int(mlflow.get_run(run_id).data.tags["n_folds"])

    # load models
    models = []
    for fold in range(n_folds):
        models.append(load_model(run_id, f"model_{fold}"))
    return models
