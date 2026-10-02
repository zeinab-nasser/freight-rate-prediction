from pathlib import Path

import numpy as np
import pandas as pd

from src.features import create_features
from src.model import load_model


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# ============================================================
# Artifact paths
# ============================================================

FINAL_PREPROCESSOR_PATH = (
    ARTIFACTS_DIR / "final_preprocessor.joblib"
)

FINAL_MODEL_PATH = (
    ARTIFACTS_DIR / "final_linear_regression.joblib"
)


# ============================================================
# Data paths
# ============================================================

TRAIN_TEST_PATH = DATA_DIR / "train-test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
DECEMBER_PATH = DATA_DIR / "december-chart-inputs.csv"

VALIDATION_OUTPUT_PATH = (
    PROJECT_ROOT / "validation_predictions.csv"
)


# ============================================================
# Load final model components
# ============================================================

def load_prediction_components():
    """
    Load the final preprocessing pipeline and trained model.

    Returns
    -------
    tuple
        (preprocessor, model)
    """

    if not FINAL_PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Final preprocessor not found: "
            f"{FINAL_PREPROCESSOR_PATH}"
        )

    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found: "
            f"{FINAL_MODEL_PATH}"
        )

    preprocessor = load_model(
        FINAL_PREPROCESSOR_PATH
    )

    model = load_model(
        FINAL_MODEL_PATH
    )

    return preprocessor, model


# ============================================================
# Generate predictions
# ============================================================

def generate_predictions(df, preprocessor, model):
    """
    Generate predictions for raw freight-rate data.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input dataframe.

    preprocessor :
        Fitted preprocessing pipeline.

    model :
        Trained regression model.

    Returns
    -------
    numpy.ndarray
        Positive predicted freight rates.
    """

    # Create engineered features
    df_features = create_features(df)

    # Transform using the fitted preprocessing pipeline
    X_processed = preprocessor.transform(
        df_features
    )

    # Generate predictions
    predictions = model.predict(
        X_processed
    )

    # Freight rates must be positive
    predictions = np.clip(
        predictions,
        1.0,
        None
    )

    return predictions


# ============================================================
# Validation predictions
# ============================================================

def predict_validation(
    preprocessor,
    model,
):
    """
    Generate predictions for the assessment validation set
    and save them using the required submission format.

    Returns
    -------
    pandas.DataFrame
        Validation predictions.
    """

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation file not found: "
            f"{VALIDATION_PATH}"
        )

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    predictions = generate_predictions(
        validation_df,
        preprocessor,
        model,
    )

    validation_output = pd.DataFrame({
        "load_id": validation_df["load_id"],
        "predicted_rate": predictions,
    })

    # Final validation checks
    if len(validation_output) != len(validation_df):
        raise ValueError(
            "Prediction count does not match "
            "validation row count."
        )

    if validation_output["load_id"].duplicated().any():
        raise ValueError(
            "Duplicate load IDs detected "
            "in validation predictions."
        )

    if validation_output["predicted_rate"].isna().any():
        raise ValueError(
            "Missing predictions detected."
        )

    if (
        validation_output["predicted_rate"] <= 0
    ).any():
        raise ValueError(
            "Non-positive predictions detected."
        )

    # Save required submission file
    validation_output.to_csv(
        VALIDATION_OUTPUT_PATH,
        index=False,
    )

    print(
        "Validation predictions saved:"
    )
    print(
        VALIDATION_OUTPUT_PATH
    )

    print(
        "Validation shape:",
        validation_output.shape
    )

    return validation_output


# ============================================================
# December market information
# ============================================================

def get_december_market_features(
    train_test_df
):
    """
    Estimate December market features using the last
    available training month (October 2025).

    Returns
    -------
    tuple
        (market_index_mean, quote_signal_mean)
    """

    train_test_df = train_test_df.copy()

    train_test_df["date"] = pd.to_datetime(
        train_test_df["date"]
    )

    october_data = train_test_df[
        train_test_df["date"].dt.month == 10
    ]

    if october_data.empty:
        raise ValueError(
            "No October training data found "
            "for December market features."
        )

    market_index = (
        october_data["market_index"]
        .mean()
    )

    quote_signal = (
        october_data["quote_signal"]
        .mean()
    )

    if pd.isna(market_index):
        raise ValueError(
            "Unable to calculate December market_index."
        )

    if pd.isna(quote_signal):
        raise ValueError(
            "Unable to calculate December quote_signal."
        )

    return market_index, quote_signal


# ============================================================
# December geographic information
# ============================================================

def get_location_coordinates(
    train_test_df,
    pickup,
    delivery,
):
    """
    Retrieve pickup and delivery coordinates from
    the development dataset.

    Returns
    -------
    dict
        Geographic coordinates for the route.
    """

    pickup_rows = train_test_df[
        train_test_df["pickup"] == pickup
    ]

    delivery_rows = train_test_df[
        train_test_df["delivery"] == delivery
    ]

    if pickup_rows.empty:
        raise ValueError(
            f"No coordinates found for pickup: {pickup}"
        )

    if delivery_rows.empty:
        raise ValueError(
            f"No coordinates found for delivery: {delivery}"
        )

    pickup_coordinates = (
        pickup_rows[
            ["pickup_lat", "pickup_lon"]
        ]
        .drop_duplicates()
    )

    delivery_coordinates = (
        delivery_rows[
            ["delivery_lat", "delivery_lon"]
        ]
        .drop_duplicates()
    )

    if len(pickup_coordinates) != 1:
        raise ValueError(
            f"Expected exactly one coordinate pair "
            f"for pickup '{pickup}', found "
            f"{len(pickup_coordinates)}."
        )

    if len(delivery_coordinates) != 1:
        raise ValueError(
            f"Expected exactly one coordinate pair "
            f"for delivery '{delivery}', found "
            f"{len(delivery_coordinates)}."
        )

    pickup_lat = pickup_coordinates.iloc[0][
        "pickup_lat"
    ]

    pickup_lon = pickup_coordinates.iloc[0][
        "pickup_lon"
    ]

    delivery_lat = delivery_coordinates.iloc[0][
        "delivery_lat"
    ]

    delivery_lon = delivery_coordinates.iloc[0][
        "delivery_lon"
    ]

    return {
        "pickup_lat": pickup_lat,
        "pickup_lon": pickup_lon,
        "delivery_lat": delivery_lat,
        "delivery_lon": delivery_lon,
    }


# ============================================================
# December predictions
# ============================================================

def predict_december(
    preprocessor,
    model,
):
    """
    Generate December 2025 predictions for the chart input
    and update the December CSV with predicted rates.

    Returns
    -------
    pandas.DataFrame
        December data with predicted_rate.
    """

    if not TRAIN_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: "
            f"{TRAIN_TEST_PATH}"
        )

    if not DECEMBER_PATH.exists():
        raise FileNotFoundError(
            f"December input file not found: "
            f"{DECEMBER_PATH}"
        )

    train_test_df = pd.read_csv(
        TRAIN_TEST_PATH
    )

    december_df = pd.read_csv(
        DECEMBER_PATH
    )

    required_columns = [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in december_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "December input is missing required "
            f"columns: {missing_columns}"
        )

    # Ensure date is datetime
    december_df["date"] = pd.to_datetime(
        december_df["date"]
    )

    # Validate December input
    if december_df.empty:
        raise ValueError(
            "December input contains no rows."
        )

    # This assessment chart uses one fixed route
    unique_pickups = (
        december_df["pickup"]
        .dropna()
        .unique()
    )

    unique_deliveries = (
        december_df["delivery"]
        .dropna()
        .unique()
    )

    if len(unique_pickups) != 1:
        raise ValueError(
            "Expected exactly one pickup location "
            "in December input."
        )

    if len(unique_deliveries) != 1:
        raise ValueError(
            "Expected exactly one delivery location "
            "in December input."
        )

    pickup = unique_pickups[0]
    delivery = unique_deliveries[0]

    # Retrieve geographic coordinates
    coordinates = get_location_coordinates(
        train_test_df,
        pickup,
        delivery,
    )

    for column, value in coordinates.items():
        december_df[column] = value

    # Retrieve last available market information
    (
        market_index,
        quote_signal,
    ) = get_december_market_features(
        train_test_df
    )

    december_df["market_index"] = (
        market_index
    )

    december_df["quote_signal"] = (
        quote_signal
    )

    # Generate predictions
    predictions = generate_predictions(
        december_df,
        preprocessor,
        model,
    )

    december_df["predicted_rate"] = (
        predictions
    )

    # Keep only the required chart columns
    december_output = december_df[
        [
            "pickup",
            "delivery",
            "distance",
            "equipment",
            "weight",
            "date",
            "predicted_rate",
        ]
    ].copy()

    # Final checks
    if len(december_output) != 31:
        raise ValueError(
            "Expected 31 December rows, found "
            f"{len(december_output)}."
        )

    if december_output[
        "predicted_rate"
    ].isna().any():
        raise ValueError(
            "Missing December predictions detected."
        )

    if (
        december_output["predicted_rate"] <= 0
    ).any():
        raise ValueError(
            "Non-positive December predictions detected."
        )

    # Save chart input
    december_output.to_csv(
        DECEMBER_PATH,
        index=False,
    )

    print(
        "December predictions saved:"
    )
    print(
        DECEMBER_PATH
    )

    print(
        "December shape:",
        december_output.shape
    )

    return december_output


# ============================================================
# Main execution
# ============================================================

def main():
    """
    Generate both required prediction files.
    """

    print("=" * 60)
    print("Freight Rate Prediction")
    print("=" * 60)

    # Load final model components
    print("\nLoading final model artifacts...")

    preprocessor, model = (
        load_prediction_components()
    )

    print(
        "Final preprocessor loaded."
    )

    print(
        "Final model loaded."
    )

    # Validation predictions
    print(
        "\nGenerating validation predictions..."
    )

    validation_output = predict_validation(
        preprocessor,
        model,
    )

    print(
        "Validation predictions:",
        len(validation_output)
    )

    # December predictions
    print(
        "\nGenerating December predictions..."
    )

    december_output = predict_december(
        preprocessor,
        model,
    )

    print(
        "December predictions:",
        len(december_output)
    )

    print("\n" + "=" * 60)
    print("Prediction pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()