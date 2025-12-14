from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


def build_model(preprocessor, random_state: int = 42) -> Pipeline:
    """Construct the full pipeline with preprocessing and regressor."""
    regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", regressor)])
