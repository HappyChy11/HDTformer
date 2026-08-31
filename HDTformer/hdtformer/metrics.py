import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    targets = np.asarray(targets).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    return {
        "r2": float(r2_score(targets, predictions)),
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
    }
