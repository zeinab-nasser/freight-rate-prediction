import joblib

from sklearn.linear_model import LinearRegression


def train_model(X_train, y_train):
    """
    Train the freight rate prediction model.
    """

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    return model


def save_model(model, path):
    """
    Save trained model to disk.
    """

    joblib.dump(
        model,
        path
    )


def load_model(path):
    """
    Load trained model from disk.
    """

    return joblib.load(path)