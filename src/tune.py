import mlflow
import optuna
import pandas as pd
from sklearn.pipeline import Pipeline

from . import config, models, params, preprocessors, utils

samplers = {
    "tpe": optuna.samplers.TPESampler(),
    "random": optuna.samplers.RandomSampler(),
}


@utils.timer
def tune(
    model: str,
    preprocessor: str,
    n_trials: int,
    timeout: float,
    data_path="",
    sampler=None,
) -> None:
    """Tune a model's hyperparameters."""
    # load data
    if not data_path:
        data_path = config.TRAIN_DATA

    train_df = pd.read_csv(data_path, index_col=config.INDEX_COL)

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

    # obtain the hyperparameter space for the model
    param_distributions = params.get_params(model)

    tags = {"model": model, "preprocessor": preprocessor, "n_folds": config.NUM_FOLDS}
    with mlflow.start_run(
        run_name=f"{model}+{preprocessor}+{config.NUM_FOLDS}",
        tags=tags,
    ):
        # create the study
        study = optuna.create_study(
            storage=config.OPTUNA_DATABASE_URL,
            sampler=samplers.get(sampler),
            direction="maximize",
            study_name=f"{model}+{preprocessor}+{sampler}+{config.NUM_FOLDS}",
            load_if_exists=True,
        )
        mlflow.set_tags({"study_name": study.study_name, "sampler": sampler})

        # hyperparameter search
        eval_metric = config.EVAL_METRICS[0]
        optuna_search = optuna.integration.OptunaSearchCV(
            pipe,
            param_distributions,
            study=study,
            n_trials=n_trials,
            timeout=timeout * 60,
            cv=config.CV_SPLITTER,
            scoring=eval_metric,
        )
        optuna_search.fit(X, y)

        # log the best parameters
        for param, value in optuna_search.best_params_.items():
            mlflow.log_param(param, value)

        # log the best score
        best_score = optuna_search.best_score_
        metrics = {eval_metric: best_score}
        if "neg_" in eval_metric:
            metrics[eval_metric.replace("neg_", "")] = -best_score
            metrics.pop(eval_metric)

        mlflow.log_metrics(metrics)

        # save the best model
        mlflow.sklearn.log_model(optuna_search.best_estimator_, "model")
