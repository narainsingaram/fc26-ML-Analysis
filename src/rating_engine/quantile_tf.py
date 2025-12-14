from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np


@dataclass
class QuantilePredictor:
    model: Any
    preprocessor: Any
    target: str

    def predict(self, X_df) -> np.ndarray:
        """Predict p10, p50, p90 for a dataframe of features."""
        # Ensure columns align with training schema if available
        try:
            expected_cols = list(getattr(self.preprocessor, "feature_names_in_", []))
            if expected_cols:
                X_df = X_df.reindex(columns=expected_cols)
        except Exception:
            pass
        X_trans = self.preprocessor.transform(X_df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        preds = self.model.predict(X_trans, verbose=0)
        return np.array(preds)


def load_quantile_predictor(model_dir: Path, target: str = "OVR") -> QuantilePredictor | None:
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        return None

    model_path = model_dir / "model.keras"
    prep_path = model_dir / "preprocessor.joblib"
    if not (model_path.exists() and prep_path.exists()):
        return None

    from tensorflow.keras.models import load_model

    preprocessor = joblib.load(prep_path)
    model = load_model(model_path, compile=False)
    return QuantilePredictor(model=model, preprocessor=preprocessor, target=target)
