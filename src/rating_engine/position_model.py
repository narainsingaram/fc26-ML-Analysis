from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import TrainingConfig
from .features import build_preprocessor, clean_dataframe, infer_feature_types, split_features_target
from .model import build_model

# Position groups used across the project.
POSITION_GROUPS = {
    "GK": {"GK"},
    "DEF": {"CB", "RB", "LB"},
    "MID": {"CM", "CDM", "CAM", "LM", "RM"},
    "ATT": {"ST", "RW", "LW"},
}


def map_position_to_group(position: str) -> str:
    for group, positions in POSITION_GROUPS.items():
        if position in positions:
            return group
    return "OTHER"


@dataclass
class GroupModelArtifact:
    model: object
    metrics: Dict[str, float]
    n_train: int
    n_test: int


@dataclass
class PositionEnsembleModel:
    """Router that dispatches to a per-position-group model."""

    group_models: Dict[str, object]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(X))
        for group in POSITION_GROUPS.keys():
            mask = X["Position"].apply(lambda p: map_position_to_group(p) == group)
            if not mask.any():
                continue
            group_model = self.group_models.get(group)
            if group_model is None:
                continue
            preds[mask] = group_model.predict(X[mask])
        return preds


def train_group_model(df: pd.DataFrame, config: TrainingConfig) -> GroupModelArtifact:
    numeric_cols, categorical_cols = infer_feature_types(df, config)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = build_model(preprocessor, random_state=config.random_state)

    X, y = split_features_target(df, config.target)
    stratify_labels = df["Position"] if df["Position"].nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_labels,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    from .evaluation import regression_metrics

    metrics = regression_metrics(y_test, y_pred)
    return GroupModelArtifact(model=model, metrics=metrics, n_train=len(X_train), n_test=len(X_test))


def train_position_ensemble(df: pd.DataFrame, config: TrainingConfig) -> Tuple[PositionEnsembleModel, Dict[str, Dict]]:
    """Train separate models per position group and return router + metrics."""
    df = df.copy()
    df["pos_group"] = df["Position"].apply(map_position_to_group)
    group_models: Dict[str, object] = {}
    group_reports: Dict[str, Dict] = {}

    for group in POSITION_GROUPS.keys():
        group_df = df[df["pos_group"] == group]
        if group_df.empty:
            continue
        cleaned = clean_dataframe(group_df, config)
        artifact = train_group_model(cleaned, config)
        group_models[group] = artifact.model
        group_reports[group] = {
            "metrics": artifact.metrics,
            "n_train": artifact.n_train,
            "n_test": artifact.n_test,
        }

    ensemble = PositionEnsembleModel(group_models=group_models)
    return ensemble, group_reports


def save_position_ensemble(path: str, ensemble: PositionEnsembleModel, reports: Dict[str, Dict]) -> None:
    joblib.dump({"ensemble": ensemble, "reports": reports}, path)


def load_position_ensemble(path: str) -> Tuple[PositionEnsembleModel, Dict[str, Dict]]:
    payload = joblib.load(path)
    return payload["ensemble"], payload.get("reports", {})
