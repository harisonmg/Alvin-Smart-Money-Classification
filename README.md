# Kaggle TPS
Template for Kaggle
[Tabular Playground Series competition](https://www.kaggle.com/competitions/?searchQuery=tabular+playground+series)
solutions

## Requirements
- Python 3.8 or greater
- Pipenv or conda package manager

## Usage
1. Create virtual environment and install dependencies
    ```shell
    # pipenv
    pipenv install --dev

    # conda
    conda env create -f environment.yml
    ```
1. Activate the virtual environment
    ```shell
    # pipenv
    pipenv shell

    # conda
    conda activate tps
    ```
1. Download the data and extract it in the `input` directory
    ```shell
    kaggle competitions download -c [competition] -p input

    # linux
    unzip "input/*.zip" -d input
    ```

1. Train a model
    ```shell
    # view train options
    python src/cli.py train --help

    # train a model
    python src/cli.py train --model [model] --preprocessor [preprocessor]
    ```

1. Make predictions on test data using the trained model. Predictions are saved
    in the `output/predictions` directory
    ```shell
    # view predict options
    python src/cli.py predict --help

    # obtain predictions
    python src/cli.py predict --run-id [run_id]
    ```

1. Submit predictions to Kaggle
    ```shell
    kaggle competitions submit -c [competition] -f output/[file] -m "Message"
    ```

## Notes
- You need a Kaggle account to download the data. See the
    [Kaggle API docs](https://www.kaggle.com/docs/api)
    for more information on creating and storing API keys.
- By default, there are 5 folds for cross validation, but that can
    be changed with the `NUM_FOLDS` environment variable.
