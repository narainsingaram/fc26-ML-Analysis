from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from rating_engine.quantile_tf import QuantilePredictor


@dataclass
class UncertaintyResult:
    p10: float
    p50: float
    p90: float
    interval_width: float
    distance_to_train: float
    is_extrapolating: bool
    is_high_uncertainty: bool
    warnings: List[str]


class UncertaintyPredictor:
    def __init__(
        self,
        quantile_predictor: QuantilePredictor,
        knn_model: NearestNeighbors,
        preprocessor: Any,
        dist_threshold: float,
        width_threshold: float,
    ):
        self.quantile_predictor = quantile_predictor
        self.knn_model = knn_model
        self.preprocessor = preprocessor
        self.dist_threshold = dist_threshold
        self.width_threshold = width_threshold

    def predict(self, X_df: pd.DataFrame) -> List[UncertaintyResult]:
        # Quantile predictions
        q_preds = self.quantile_predictor.predict(X_df)  # (N, 3)

        # Distance calculations
        # Align columns
        try:
            expected_cols = list(getattr(self.preprocessor, "feature_names_in_", []))
            if expected_cols:
                # Add missing cols with 0/Nan and reorder
                for c in expected_cols:
                    if c not in X_df.columns:
                        X_df[c] = 0 # simplified handling for missing cols in inference
                X_ordered = X_df[expected_cols]
            else:
                X_ordered = X_df
        except Exception:
            X_ordered = X_df

        X_trans = self.preprocessor.transform(X_ordered)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()

        # Find distance to nearest neighbors (we take mean of k=5)
        # Note: knn_model.kneighbors returns (distances, indices)
        # If n_neighbors was set to 5 during fit/init, we typically query 5 here.
        dists, _ = self.knn_model.kneighbors(X_trans)
        mean_dists = dists.mean(axis=1)

        results = []
        for i in range(len(X_df)):
            p10, p50, p90 = q_preds[i]
            width = p90 - p10
            dist = mean_dists[i]
            
            # Checks
            is_extrapolating = bool(dist > self.dist_threshold)
            is_high_uncertainty = bool(width > self.width_threshold)
            
            warnings = []
            if is_extrapolating:
                warnings.append(f"High risk: Input data is unlike training set (Distance {dist:.2f} > {self.dist_threshold:.2f})")
            if is_high_uncertainty:
                warnings.append(f"Low confidence: Model predicts a wide range of outcomes ({width:.1f} spread)")

            results.append(
                UncertaintyResult(
                    p10=float(p10),
                    p50=float(p50),
                    p90=float(p90),
                    interval_width=float(width),
                    distance_to_train=float(dist),
                    is_extrapolating=is_extrapolating,
                    is_high_uncertainty=is_high_uncertainty,
                    warnings=warnings,
                )
            )
        return results


def load_uncertainty_predictor(model_dir: Path) -> UncertaintyPredictor | None:
    from rating_engine.quantile_tf import load_quantile_predictor as load_q_pred
    
    # Load components
    quantile_model = load_q_pred(model_dir)
    if not quantile_model:
        return None

    knn_path = model_dir / "uncertainty_knn.joblib"
    prep_path = model_dir / "uncertainty_preprocessor.joblib"
    meta_path = model_dir / "uncertainty_meta.joblib"

    if not (knn_path.exists() and prep_path.exists() and meta_path.exists()):
        return None

    knn_model = joblib.load(knn_path)
    preprocessor = joblib.load(prep_path)
    meta = joblib.load(meta_path)

    return UncertaintyPredictor(
        quantile_predictor=quantile_model,
        knn_model=knn_model,
        preprocessor=preprocessor,
        dist_threshold=meta["dist_threshold"],
        width_threshold=meta["width_threshold"],
    )
