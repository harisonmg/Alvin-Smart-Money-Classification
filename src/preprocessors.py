from feature_engine.datetime import DatetimeFeatures
from feature_engine.transformation import LogTransformer
from sklearn import compose, impute, pipeline, preprocessing
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

vectorizers = {
    "count": text.CountVectorizer(stop_words="english"),
    "tfidf": text.TfidfVectorizer(stop_words="english"),
}


gender_pipe = pipeline.Pipeline(
    [("encoder", encoders["ordinal"]), ("imputer", imputers["mode"])]
)


preprocessors = {
    "custom_1": compose.make_column_transformer(
        (DatetimeFeatures(features_to_extract=DATETIME_FEATURES), ["PURCHASED_AT"]),
        (gender_pipe, ["USER_GENDER"]),
        (
            IdentityTransformer(),
            ["IS_PURCHASE_PAID_VIA_MPESA_SEND_MONEY", "USER_HOUSEHOLD"],
        ),
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    # add discretized income and purchase value to `custom_1`
    "custom_2": compose.make_column_transformer(
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
    # add discretized purchase value and log transformed income to `custom_1`
    "custom_3": compose.make_column_transformer(
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
    # add ordinal encoded merchant name and user id to `custom_1`
    "custom_4": compose.make_column_transformer(
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
    # add ordinal encoded merchant name and user id to `custom_2`
    "custom_5": compose.make_column_transformer(
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
    # add ordinal encoded merchant name and user id to `custom_3`
    "custom_6": compose.make_column_transformer(
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
    "name_count": compose.make_column_transformer(
        (vectorizers["count"], "MERCHANT_NAME"),
        sparse_threshold=0,
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
    "name_tfidf": compose.make_column_transformer(
        (vectorizers["tfidf"], "MERCHANT_NAME"),
        sparse_threshold=0,
        n_jobs=config.N_JOBS,
        verbose=config.VERBOSE,
    ),
}
