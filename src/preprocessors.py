from feature_engine.datetime import DatetimeFeatures
from feature_engine.transformation import LogTransformer
from sklearn import compose, decomposition, impute, pipeline, preprocessing
from sklearn.feature_extraction import text
from sklego.preprocessing import IdentityTransformer

from . import config

DATETIME_FEATURES = ["month", "day_of_month", "day_of_week", "hour"]

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

decomposers = {
    "nmf": decomposition.NMF(),
    "truncated_svd": decomposition.TruncatedSVD(),
}

vectorizers = {
    "count": text.CountVectorizer(stop_words="english"),
    "tfidf": text.TfidfVectorizer(stop_words="english"),
}


gender_pipe = pipeline.Pipeline(
    [("encoder", encoders["ordinal"]), ("imputer", imputers["mode"])],
    verbose=config.VERBOSITY,
)

text_pipelines = {
    "count_nmf": pipeline.Pipeline(
        [
            ("vectorizer", vectorizers["count"]),
            ("decomposer", decomposers["nmf"]),
        ],
        verbose=config.VERBOSITY,
    ),
    "count_truncated_svd": pipeline.Pipeline(
        [
            ("vectorizer", vectorizers["count"]),
            ("decomposer", decomposers["truncated_svd"]),
        ],
        verbose=config.VERBOSITY,
    ),
    "tfidf_nmf": pipeline.Pipeline(
        [
            ("vectorizer", vectorizers["tfidf"]),
            ("decomposer", decomposers["nmf"]),
        ],
        verbose=config.VERBOSITY,
    ),
    "tfidf_truncated_svd": pipeline.Pipeline(
        [
            ("vectorizer", vectorizers["tfidf"]),
            ("decomposer", decomposers["truncated_svd"]),
        ],
        verbose=config.VERBOSITY,
    ),
}


preprocessors = {
    "c1": compose.make_column_transformer(
        (DatetimeFeatures(features_to_extract=DATETIME_FEATURES), ["PURCHASED_AT"]),
        (gender_pipe, ["USER_GENDER"]),
        (
            IdentityTransformer(),
            ["IS_PURCHASE_PAID_VIA_MPESA_SEND_MONEY", "USER_HOUSEHOLD"],
        ),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    # add discretized income and purchase value to `c1`
    "c2": compose.make_column_transformer(
        (DatetimeFeatures(features_to_extract=DATETIME_FEATURES), ["PURCHASED_AT"]),
        (gender_pipe, ["USER_GENDER"]),
        (
            IdentityTransformer(),
            ["IS_PURCHASE_PAID_VIA_MPESA_SEND_MONEY", "USER_HOUSEHOLD"],
        ),
        (
            preprocessing.KBinsDiscretizer(encode="ordinal"),
            ["PURCHASE_VALUE", "USER_INCOME"],
        ),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    # add discretized purchase value and log transformed income to `c1`
    "c3": compose.make_column_transformer(
        (DatetimeFeatures(features_to_extract=DATETIME_FEATURES), ["PURCHASED_AT"]),
        (gender_pipe, ["USER_GENDER"]),
        (
            IdentityTransformer(),
            ["IS_PURCHASE_PAID_VIA_MPESA_SEND_MONEY", "USER_HOUSEHOLD"],
        ),
        (preprocessing.KBinsDiscretizer(encode="ordinal"), ["PURCHASE_VALUE"]),
        (LogTransformer(), ["USER_INCOME"]),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    # add ordinal encoded merchant name and user id to `c1`
    "c4": compose.make_column_transformer(
        (DatetimeFeatures(features_to_extract=DATETIME_FEATURES), ["PURCHASED_AT"]),
        (gender_pipe, ["USER_GENDER"]),
        (
            IdentityTransformer(),
            ["IS_PURCHASE_PAID_VIA_MPESA_SEND_MONEY", "USER_HOUSEHOLD"],
        ),
        (encoders["ordinal"], ["MERCHANT_NAME", "USER_ID"]),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    # add ordinal encoded merchant name and user id to `c2`
    "c5": compose.make_column_transformer(
        (DatetimeFeatures(features_to_extract=DATETIME_FEATURES), ["PURCHASED_AT"]),
        (gender_pipe, ["USER_GENDER"]),
        (
            IdentityTransformer(),
            ["IS_PURCHASE_PAID_VIA_MPESA_SEND_MONEY", "USER_HOUSEHOLD"],
        ),
        (
            preprocessing.KBinsDiscretizer(encode="ordinal"),
            ["PURCHASE_VALUE", "USER_INCOME"],
        ),
        (encoders["ordinal"], ["MERCHANT_NAME", "USER_ID"]),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    # add ordinal encoded merchant name and user id to `c3`
    "c6": compose.make_column_transformer(
        (DatetimeFeatures(features_to_extract=DATETIME_FEATURES), ["PURCHASED_AT"]),
        (gender_pipe, ["USER_GENDER"]),
        (
            IdentityTransformer(),
            ["IS_PURCHASE_PAID_VIA_MPESA_SEND_MONEY", "USER_HOUSEHOLD"],
        ),
        (preprocessing.KBinsDiscretizer(encode="ordinal"), ["PURCHASE_VALUE"]),
        (LogTransformer(), ["USER_INCOME"]),
        (encoders["ordinal"], ["MERCHANT_NAME", "USER_ID"]),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    "n1": compose.make_column_transformer(
        (vectorizers["count"], "MERCHANT_NAME"),
        sparse_threshold=0,
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    "n2": compose.make_column_transformer(
        (vectorizers["tfidf"], "MERCHANT_NAME"),
        sparse_threshold=0,
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    "n3": compose.make_column_transformer(
        (text_pipelines["count_nmf"], "MERCHANT_NAME"),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    "n4": compose.make_column_transformer(
        (text_pipelines["count_truncated_svd"], "MERCHANT_NAME"),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    "n5": compose.make_column_transformer(
        (text_pipelines["tfidf_nmf"], "MERCHANT_NAME"),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    "n6": compose.make_column_transformer(
        (text_pipelines["tfidf_truncated_svd"], "MERCHANT_NAME"),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
}
