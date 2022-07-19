from sklearn import compose, impute, pipeline, preprocessing

from . import config

NUMERIC_DTYPES = ["int64", "float64"]


def get_numeric_cols():
    if config.CONTINUOUS_FEATURES:
        return config.CONTINUOUS_FEATURES
    return compose.make_column_selector(dtype_include=NUMERIC_DTYPES)


def get_categorical_cols():
    if config.DISCRETE_FEATURES:
        return config.DISCRETE_FEATURES
    return compose.make_column_selector(dtype_exclude=NUMERIC_DTYPES)


imputers = {
    "constant": impute.SimpleImputer(strategy="constant", fill_value="unknown"),
    "knn": impute.KNNImputer(),
    "mean": impute.SimpleImputer(),
    "median": impute.SimpleImputer(strategy="median"),
    "mode": impute.SimpleImputer(strategy="most_frequent"),
}

encoders = {
    "one_hot": preprocessing.OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=0.01
    ),
    "ordinal": preprocessing.OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-999
    ),
}

scalers = {
    "max_abs": preprocessing.MaxAbsScaler(),
    "min_max": preprocessing.MinMaxScaler(),
    "standard": preprocessing.StandardScaler(),
}

numeric_pipelines = {
    "linear": pipeline.Pipeline(
        [("standard_scaler", scalers["standard"]), ("mean_imputer", imputers["mean"])],
        verbose=config.VERBOSITY,
    ),
    "tree": pipeline.Pipeline(
        [("mean_imputer", imputers["mean"])], verbose=config.VERBOSITY
    ),
}

categorical_pipelines = {
    "linear": pipeline.Pipeline(
        [("one_hot_encoder", encoders["one_hot"])], verbose=config.VERBOSITY
    ),
    "tree": pipeline.Pipeline(
        [("mode_imputer", imputers["mode"]), ("ordinal_encoder", encoders["ordinal"])],
        verbose=config.VERBOSITY,
    ),
}

preprocessors = {
    "lin_num": compose.make_column_transformer(
        (numeric_pipelines["linear"], get_numeric_cols()),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    "tree_num": compose.make_column_transformer(
        (numeric_pipelines["tree"], get_numeric_cols()),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    "lin_cat": compose.make_column_transformer(
        (categorical_pipelines["linear"], get_categorical_cols()),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    "tree_cat": compose.make_column_transformer(
        (categorical_pipelines["tree"], get_categorical_cols()),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    "lin_all": compose.make_column_transformer(
        (numeric_pipelines["linear"], get_numeric_cols()),
        (categorical_pipelines["linear"], get_categorical_cols()),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    "tree_all": compose.make_column_transformer(
        (numeric_pipelines["tree"], get_numeric_cols()),
        (categorical_pipelines["tree"], get_categorical_cols()),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
}
