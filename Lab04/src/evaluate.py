"""Regression metrics in the original target units."""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def metrics(actual, predicted):
    mse = float(mean_squared_error(actual, predicted))
    return dict(mse=mse, rmse=float(np.sqrt(mse)), mae=float(mean_absolute_error(actual, predicted)), r2=float(r2_score(actual, predicted)))
