import argparse

from .models import models
from .params import param_distributions
from .predict import predict
from .preprocessors import preprocessors
from .train import train
from .tune import samplers, tune
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
    parse_tune(subparsers)

    # parse the arguments from the command line and call the callback function
    args = parser.parse_args()
    args.func(args)


def train_callback(args: argparse.Namespace):
    """Callback function for the train command"""
    train(model=args.model, preprocessor=args.preprocessor, data_path=args.file)


def tune_callback(args: argparse.Namespace):
    """Callback function for the tune command"""
    tune(
        model=args.model,
        preprocessor=args.preprocessor,
        data_path=args.file,
        n_trials=args.trials,
        timeout=args.timeout,
        sampler=args.sampler,
    )


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


def parse_tune(subparsers: argparse.ArgumentParser):
    """Subparser for the tune command"""
    parser_tune = subparsers.add_parser("tune", help="tune a model's hyperparameters")
    parser_tune.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        choices=param_distributions.keys(),
    )
    parser_tune.add_argument(
        "-p",
        "--preprocessor",
        type=str,
        required=True,
        choices=preprocessors.keys(),
    )
    parser_tune.add_argument(
        "-f", "--file", type=str, help="path to the file containing the data"
    )
    parser_tune.add_argument(
        "--trials", type=int, default=5, help="number of trials to run"
    )
    parser_tune.add_argument(
        "--timeout",
        type=float,
        default=10,
        help="maximum number of minutes to tune the model",
    )
    parser_tune.add_argument(
        "--sampler",
        type=str,
        default="random",
        choices=samplers.keys(),
        help="sampler to use",
    )

    # add the callback for the tune command
    parser_tune.set_defaults(func=tune_callback)


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
