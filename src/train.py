import logging

import mlflow
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline

from . import config, models, utils
from .preprocessors import preprocessor

# logger
logger = logging.getLogger(__name__)


def summarize_metrics(metrics: dict) -> pd.DataFrame:
    """Summarize metrics from cross-validation."""
    metrics_df = pd.DataFrame(metrics)

    # convert negated metrics to positive
    negative_metrics = metrics_df.columns.str.contains("neg_")
    metrics_df.loc[:, negative_metrics] = -metrics_df.loc[:, negative_metrics]

    # rename columns
    metrics_df.columns = metrics_df.columns.str.replace("neg_", "")

    # obtain mean and std
    summary = metrics_df.describe().round(5)
    return summary.loc[["mean", "std"]]


def log_metrics(metrics: dict) -> None:
    """Log cross-validation metrics."""
    summary = summarize_metrics(metrics)

    # get the current run id
    run_id = mlflow.active_run().info.run_id
    logger.info(f"Cross validation results\nfor run {run_id!r}:\n{summary.T}")

    # log metrics to MLflow
    for metric, values in summary.iteritems():
        mlflow.log_metric(f"{metric}_mean", values["mean"])
        mlflow.log_metric(f"{metric}_std", values["std"])


@utils.timer
def save_models(models: list) -> None:
    """Save models as mlflow artifacts."""
    for fold, model in enumerate(models):
        mlflow.sklearn.log_model(model, f"model_{fold}")


@utils.timer
def train(model: str, data_path="") -> None:
    """Train model."""
    # load data
    if not data_path:
        data_path = config.TRAIN_DATA

    train_df = pd.read_csv(data_path, index_col=config.INDEX_COL)

    # separate features from target
    X = train_df.drop(config.TARGET_COL, axis=1)
    y = train_df[config.TARGET_COL]

    # create pipeline
    pipe = make_pipeline(
        preprocessor,
        models.models[model],
        verbose=config.VERBOSE,
    )

    tags = {"model": model, "n_folds": config.NUM_FOLDS}
    with mlflow.start_run(
        experiment_id=config.MLFLOW_EXPERIMENT_ID,
        run_name=f"{model}, {config.NUM_FOLDS} folds",
        tags=tags,
    ):
        # log model parameters
        for param, value in pipe.get_params().items():
            mlflow.log_param(param, value)

        # cross validation
        cv_results = cross_validate(
            pipe,
            X,
            y,
            scoring=config.EVAL_METRICS,
            cv=config.CV_SPLITTER,
            n_jobs=config.N_JOBS,
            verbose=config.VERBOSITY,
            return_estimator=True,
        )

        # summarize CV results
        estimators = cv_results.pop("estimator")

        # log metrics
        log_metrics(cv_results)

        # save the models
        save_models(estimators)
