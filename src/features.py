import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two geographic points.

    Parameters
    ----------
    lat1, lon1 : array-like
        Pickup latitude and longitude.
    lat2, lon2 : array-like
        Delivery latitude and longitude.

    Returns
    -------
    array-like
        Geographic distance in miles.
    """
    R = 3958.8  # Earth radius in miles

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))


def create_features(df):
    """
    Create engineered features required by the freight rate model.

    The function does not modify the original DataFrame.
    """
    df = df.copy()

    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])

    # Date features
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # Geographic difference features
    df["lat_diff"] = (
        df["delivery_lat"] - df["pickup_lat"]
    )

    df["lon_diff"] = (
        df["delivery_lon"] - df["pickup_lon"]
    )

    # Geographic distance
    df["geographic_distance"] = haversine_distance(
        df["pickup_lat"],
        df["pickup_lon"],
        df["delivery_lat"],
        df["delivery_lon"],
    )

    # Route efficiency
    df["route_efficiency"] = (
        df["distance"] / df["geographic_distance"]
    )

    # Log-transformed route efficiency
    df["log_route_efficiency"] = np.log1p(
        df["route_efficiency"]
    )

    # Weight-based feature
    df["weight_per_mile"] = (
        df["weight"].abs() / df["distance"]
    )

    return df