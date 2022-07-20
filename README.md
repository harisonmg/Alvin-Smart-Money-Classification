# Alvin Smart Money Classification
Solution for the
[Alvin Smart Money Classification](https://zindi.africa/competitions/alvin-smart-money-management-classification-challenge)
challenge

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
    conda activate alvin-smcc
    ```
1. Download the data and extract it in the `input` directory
    ```shell
    # extracting the data in linux
    unzip "input/*.zip"
    ```

1. Train a model
    ```shell
    # view train options
    python src/cli.py train --help

    # train a model
    python src/cli.py train --model [model]
    ```

1. Make predictions on test data using the trained model. Predictions are saved
    in the `output/predictions` directory
    ```shell
    # view predict options
    python src/cli.py predict --help

    # obtain predictions
    python src/cli.py predict --run-id [run_id]
    ```

## Notes
- By default, there are 5 folds for cross validation, but that can
    be changed with the `NUM_FOLDS` environment variable
- Verbosity can be changed with the `VERBOSITY` environment variable
- Environment variables can be set in the `.env` file
