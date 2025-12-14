from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd
from sklearn.inspection import permutation_importance


def get_feature_names(model) -> List[str]:
    """Return transformed feature names from a fitted pipeline."""
    preprocessor = model.named_steps["preprocessor"]
    try:
        names = preprocessor.get_feature_names_out()
        return list(names)
    except Exception:
        # Fallback to positional feature ids.
        n_features = model.named_steps["model"].n_features_in_
        return [f"feature_{i}" for i in range(n_features)]


def permutation_importance_summary(
    model, X, y, n_repeats: int = 5, random_state: int = 42, n_jobs: int = 1
) -> pd.DataFrame:
    """Compute permutation importance on held-out data."""
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
    )
    names = get_feature_names(model)
    n = min(len(names), len(result.importances_mean))
    names = names[:n]
    df = pd.DataFrame(
        {
            "feature": names,
            "importance_mean": result.importances_mean[:n],
            "importance_std": result.importances_std[:n],
        }
    )
    return df.sort_values(by="importance_mean", ascending=False).reset_index(drop=True)


def save_importances(df: pd.DataFrame, path: Path, top_k: int | None = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = df.head(top_k) if top_k else df
    output.to_csv(path, index=False)
