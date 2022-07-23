import argparse

from .models import models
from .predict import predict
from .preprocessors import preprocessors
from .train import train
from .utils import configure_mlflow


@configure_mlflow
def main():
    """Main function for the CLI"""
    # create the main parser
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True, help="action help")

    # create a subparser for each subcommand
    parse_train(subparsers)
    parse_predict(subparsers)

    # parse the arguments from the command line and call the callback function
    args = parser.parse_args()
    args.func(args)


def train_callback(args: argparse.Namespace):
    """Callback function for the train command"""
    train(model=args.model, preprocessor=args.preprocessor, data_path=args.file)


def predict_callback(args: argparse.Namespace):
    """Callback function for the predict command"""
    predict(run_id=args.run_id, data_path=args.file, proba=args.proba)


def parse_train(subparsers: argparse.ArgumentParser):
    """Subparser for the train command"""
    parser_train = subparsers.add_parser("train", help="train a model")
    parser_train.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        choices=models.keys(),
    )
    parser_train.add_argument(
        "-p",
        "--preprocessor",
        type=str,
        required=True,
        choices=preprocessors.keys(),
    )
    parser_train.add_argument(
        "-f", "--file", type=str, help="path to the file containing the data"
    )

    # add the callback for the train command
    parser_train.set_defaults(func=train_callback)


def parse_predict(subparsers: argparse.ArgumentParser):
    """Subparser for the predict command"""
    # create the subparser for the predict command
    parser_predict = subparsers.add_parser("predict", help="predict on a dataset")
    parser_predict.add_argument(
        "-r",
        "--run-id",
        type=str,
        required=True,
        help="MLflow run id",
    )
    parser_predict.add_argument(
        "-p",
        "--proba",
        action="store_true",
        help="obtain probabilities instead of class labels",
    )
    parser_predict.add_argument(
        "-f", "--file", type=str, help="path to the file containing the data"
    )

    # add the callback for the predict command
    parser_predict.set_defaults(func=predict_callback)


if __name__ == "__main__":
    main()
