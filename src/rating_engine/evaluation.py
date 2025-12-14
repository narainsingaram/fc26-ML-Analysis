from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute common regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_metrics(metrics: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(metrics, f, indent=2)


def summarize_predictions(y_true, y_pred):
    """Return simple error distribution stats."""
    residuals = np.array(y_pred) - np.array(y_true)
    return {
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "residual_p90": float(np.percentile(residuals, 90)),
        "residual_p10": float(np.percentile(residuals, 10)),
    }
