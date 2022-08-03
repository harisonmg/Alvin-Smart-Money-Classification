import logging

import mlflow
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from . import config, models, preprocessors, utils

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


@utils.timer
def log_metrics(metrics: dict) -> None:
    """Log cross-validation metrics."""
    summary = summarize_metrics(metrics)

    # get the current run id
    run_id = mlflow.active_run().info.run_id
    logger.info(f"Cross validation results\nfor run {run_id!r}:\n{summary.T}")

    # log metrics to MLflow
    flat_summary = pd.json_normalize(summary.to_dict())
    mlflow.log_metrics(flat_summary.loc[0].to_dict())


@utils.timer
def save_models(models: list) -> None:
    """Save models as MLflow artifacts."""
    for fold, model in enumerate(models):
        mlflow.sklearn.log_model(model, f"model_{fold}")


@utils.timer
def train(model: str, preprocessor: str, data_path="") -> None:
    """Train model."""
    # load data
    if not data_path:
        data_path = config.TRAIN_DATA

    train_df = pd.read_csv(
        data_path, index_col=config.INDEX_COL, parse_dates=config.DATETIME_COLS
    )

    # separate features from target
    X = train_df.drop(config.TARGET_COL, axis=1)
    y = train_df[config.TARGET_COL]

    # create pipeline
    pipe = Pipeline(
        [
            (preprocessor, preprocessors.preprocessors[preprocessor]),
            (model, models.models[model]),
        ],
        verbose=config.VERBOSE,
    )

    tags = {"model": model, "preprocessor": preprocessor, "n_folds": config.NUM_FOLDS}
    with mlflow.start_run(
        run_name=f"{model}+{preprocessor}+{config.NUM_FOLDS}",
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
            return_train_score=True,
        )
        estimators = cv_results.pop("estimator")

        # log metrics
        log_metrics(cv_results)

        # save the models
        save_models(estimators)
