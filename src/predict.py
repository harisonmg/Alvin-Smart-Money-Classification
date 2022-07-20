from pathlib import Path

import mlflow
import pandas as pd

from . import config, utils


@utils.timer
def save_predictions(predictions: pd.DataFrame, file_name: str) -> None:
    """Save predictions"""
    # create folder for saving predictions
    predictions_path = config.OUTPUT_DIR / "predictions"
    if not predictions_path.exists():
        predictions_path.mkdir()

    # save predictions
    file = predictions_path / file_name
    predictions.to_csv(file, index=False)


@utils.timer
def predict(run_id: str, data_path="", proba=False, save_preds=True) -> None:
    # load data
    if not data_path:
        data_path = config.TEST_DATA
    else:
        data_path = Path(data_path)

    test_df = pd.read_csv(
        data_path, index_col=config.INDEX_COL, parse_dates=config.DATETIME_COLS
    )

    # get the number of folds for this run
    n_folds = int(mlflow.get_run(run_id).data.tags["n_folds"])

    # obtain predictions
    predictions = None
    for fold in range(n_folds):
        # load model
        logged_model = f"runs:/{run_id}/model_{fold}"
        estimator = mlflow.sklearn.load_model(logged_model)
        test_preds = estimator.predict_proba(test_df)

        if predictions is None:
            predictions = test_preds
        else:
            predictions += test_preds

    # average predictions and create a dataframe
    predictions /= n_folds
    predictions_df = pd.DataFrame(
        predictions, columns=estimator.classes_, index=test_df.index
    )

    # format predictions depending on whether we want probabilities or classes
    if proba:
        # get the probability of the positive class for binary classification
        if predictions_df.shape[1] == 2:
            predictions_df = predictions_df.iloc[:, 1].rename(config.TARGET_COL)
    else:
        predictions_df = predictions_df.idxmax(axis=1).rename(config.TARGET_COL)

    # reset the index
    predictions_df = predictions_df.reset_index()

    # save predictions
    if save_preds:
        save_predictions(predictions_df, f"{run_id}_{data_path.stem}.csv")

    # return predictions
    return predictions_df
