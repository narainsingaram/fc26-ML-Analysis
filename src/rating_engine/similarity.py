from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from .config import TrainingConfig
from .features import clean_dataframe, infer_feature_types


@dataclass
class SimilarityIndex:
    vectors: np.ndarray
    ids: np.ndarray
    meta: pd.DataFrame
    scaler: StandardScaler
    numeric_cols: List[str]

    def query(self, player_id: int, top_k: int = 10) -> List[Dict]:
        """Return top_k most similar players (excluding the query player)."""
        matches = np.where(self.ids == player_id)[0]
        if len(matches) == 0:
            raise KeyError(f"Player ID {player_id} not found in index")
        idx = matches[0]
        base_vec = self.vectors[idx].reshape(1, -1)
        sims = cosine_similarity(base_vec, self.vectors)[0]
        order = np.argsort(-sims)
        results = []
        for j in order:
            if self.ids[j] == player_id:
                continue
            results.append(
                {
                    "id": int(self.ids[j]),
                    "similarity": float(sims[j]),
                    "meta": self.meta.iloc[j].to_dict(),
                }
            )
            if len(results) >= top_k:
                break
        return results


def build_similarity_index(df: pd.DataFrame, config: TrainingConfig) -> SimilarityIndex:
    """Create normalized numeric vectors and supporting metadata."""
    cleaned = clean_dataframe(df, config)
    numeric_cols, _ = infer_feature_types(cleaned, config)
    numeric_df = cleaned[numeric_cols].copy()
    numeric_df = numeric_df.fillna(numeric_df.mean())
    scaler = StandardScaler()
    vectors = scaler.fit_transform(numeric_df)

    meta_cols = ["ID", "Name", "Position", "OVR", "Team", "League", "card"]
    meta = df[meta_cols].copy()
    return SimilarityIndex(
        vectors=vectors,
        ids=meta["ID"].to_numpy(),
        meta=meta,
        scaler=scaler,
        numeric_cols=numeric_cols,
    )
