import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer,
)


# ============================================================
# Feature groups
# ============================================================

CATEGORICAL_FEATURES = [
    "pickup",
    "delivery",
    "equipment",
]

WEIGHT_FEATURES = [
    "weight",
]

NUMERIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "market_index",
    "quote_signal",
    "month",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "lat_diff",
    "lon_diff",
    "geographic_distance",
    "log_route_efficiency",
    "weight_per_mile",
]


# ============================================================
# Preprocessing pipelines
# ============================================================

weight_pipeline = Pipeline([
    (
        "abs",
        FunctionTransformer(
            np.abs,
            feature_names_out="one-to-one",
        ),
    ),
    (
        "imputer",
        SimpleImputer(strategy="median"),
    ),
    (
        "scaler",
        StandardScaler(),
    ),
])


numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median"),
    ),
    (
        "scaler",
        StandardScaler(),
    ),
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent"),
    ),
    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
        ),
    ),
])


# ============================================================
# Build preprocessor
# ============================================================

def build_preprocessor():
    """
    Build and return the complete preprocessing pipeline.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "weight",
                weight_pipeline,
                WEIGHT_FEATURES,
            ),
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor