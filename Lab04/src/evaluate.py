
"""Prediction and regression evaluation metrics."""

import torch
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)



def predict(model, X):
    """Generate predictions using a trained model."""

    was_training = model.training
    model.eval()

    device = next(model.parameters()).device

    X_tensor = torch.as_tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    try:
        with torch.no_grad():
            predictions = model(X_tensor)
    finally:
        if was_training:
            model.train()

    return predictions.cpu().numpy()



def metrics(actual, predicted):
    """Calculate regression metrics."""

    mse = float(mean_squared_error(actual, predicted))

    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }
