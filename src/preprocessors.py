import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

UNTRANSFORMED_COLS = ["IS_PURCHASE_PAID_VIA_MPESA_SEND_MONEY", "USER_HOUSEHOLD"]


def featurize_ts(series: pd.Series) -> pd.DataFrame:
    """Extract features from a timestamp column"""
    df = pd.DataFrame()
    col_prefix = series.name
    df[f"{col_prefix}_month"] = series.dt.month
    df[f"{col_prefix}_day"] = series.dt.day
    df[f"{col_prefix}_weekday"] = series.dt.weekday
    df[f"{col_prefix}_hour"] = series.dt.hour
    return df


def log_transform(arr: np.ndarray) -> np.ndarray:
    return np.expand_dims(np.log(arr), axis=1)


encoder_pipe = Pipeline(
    [
        (
            "encode",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-999),
        ),
    ]
)
imputer_pipe = Pipeline(
    [
        ("expand_dims", FunctionTransformer(np.expand_dims, kw_args={"axis": 1})),
        ("impute", SimpleImputer(strategy="constant", fill_value=-999)),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("merch_name_vec", CountVectorizer(stop_words="english"), "MERCHANT_NAME"),
        ("purchased_ts", FunctionTransformer(featurize_ts), "PURCHASED_AT"),
        ("identity", FunctionTransformer(lambda x: x * 1), UNTRANSFORMED_COLS),
        ("log_purchase", FunctionTransformer(log_transform), "PURCHASE_VALUE"),
        ("log_income", FunctionTransformer(log_transform), "USER_INCOME"),
        ("encode", encoder_pipe, ["USER_ID", "USER_GENDER"]),
        ("impute_age", imputer_pipe, "USER_AGE"),
    ],
    n_jobs=-1,
)
