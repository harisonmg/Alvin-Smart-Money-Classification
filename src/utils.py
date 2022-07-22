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
