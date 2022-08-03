import optuna.distributions as dist

param_distributions = {
    "xgb": {
        "n_estimators": dist.IntUniformDistribution(10, 5000),
        "learning_rate": dist.LogUniformDistribution(0.01, 0.1),
        "colsample_bytree": dist.UniformDistribution(0.1, 1.0),
        "max_depth": dist.IntUniformDistribution(1, 50),
        "subsample": dist.UniformDistribution(0.1, 1.0),
        "reg_alpha": dist.LogUniformDistribution(1e-9, 100.0),
        "reg_lambda": dist.LogUniformDistribution(1e-9, 100.0),
    },
    "cb": {
        "iterations": dist.IntUniformDistribution(10, 5000),
        "learning_rate": dist.LogUniformDistribution(0.01, 0.1),
        "depth": dist.IntUniformDistribution(1, 8),
        "random_strength": dist.LogUniformDistribution(1e-9, 10.0),
        "bagging_temperature": dist.LogUniformDistribution(1e-9, 1.0),
        "border_count": dist.IntUniformDistribution(1, 255),
        "l2_leaf_reg": dist.IntUniformDistribution(2, 30),
    },
    "lgb": {
        "n_estimators": dist.IntUniformDistribution(10, 1000),
        "learning_rate": dist.LogUniformDistribution(0.01, 0.1),
        "feature_fraction": dist.UniformDistribution(0.1, 1.0),
        "min_data_in_leaf": dist.IntUniformDistribution(0, 300),
        "subsample": dist.UniformDistribution(0.01, 1.0),
        "reg_alpha": dist.LogUniformDistribution(1e-8, 10.0),
        "reg_lambda": dist.LogUniformDistribution(1e-8, 10.0),
    },
    "hgb": {
        "learning_rate": dist.LogUniformDistribution(0.01, 1.0),
        "max_iter": dist.IntUniformDistribution(10, 10000),
        "max_depth": dist.IntUniformDistribution(2, 12),
        "min_samples_leaf": dist.IntUniformDistribution(2, 300),
        "l2_regularization": dist.LogUniformDistribution(0.01, 100.0),
        "max_bins": dist.IntUniformDistribution(32, 255),
    },
}


def get_params(model_name: str) -> dict:
    """Get the hyperparameter space for the given model."""
    try:
        param_dist = param_distributions[model_name]
    except KeyError:
        raise ValueError(f"Hyperparameters for {model_name!r} not found")

    # append the model name to the parameter space keys
    param_dist = {f"{model_name}__{key}": val for key, val in param_dist.items()}
    return param_dist
