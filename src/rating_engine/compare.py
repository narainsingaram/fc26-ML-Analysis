from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from .config import TrainingConfig
from .data import load_player_data
from .features import clean_dataframe
from .explain import get_feature_names


@dataclass
class PlayerComparison:
    player_a_name: str
    player_b_name: str
    pred_a: float
    pred_b: float
    delta: float
    top_reasons: List[Dict[str, float]]
    natural_language_summary: str
    feature_table: pd.DataFrame
    what_if: List[Dict[str, float]]


def _aggregate_deltas(feature_names: List[str], deltas: np.ndarray) -> pd.Series:
    """Aggregate one-hot columns back to their base feature name."""
    base_names = []
    for name in feature_names:
        if "__" in name:
            _, remainder = name.split("__", 1)
        else:
            remainder = name
        base = remainder.split("_", 1)[0]
        base_names.append(base)
    series = pd.Series(deltas, index=base_names)
    return series.groupby(level=0).sum().sort_values(key=lambda s: s.abs(), ascending=False)


def _summarize_reasons(
    aggregated: pd.Series,
    row_a: pd.Series,
    row_b: pd.Series,
    delta: float,
    top_k: int = 8,
) -> List[Dict[str, float]]:
    def pick_value(row: pd.Series, key: str):
        if key in row:
            return row[key]
        # fallback: match columns that start with the key (ignoring spaces/underscore case)
        target = key.lower().replace(" ", "")
        for col in row.index:
            col_norm = str(col).lower().replace(" ", "").replace("__", "_")
            if col_norm.startswith(target):
                return row[col]
        return None

    top = aggregated.head(top_k)
    reasons: List[Dict[str, float]] = []
    for feat, impact in top.items():
        val_a = pick_value(row_a, feat)
        val_b = pick_value(row_b, feat)
        stat_delta = None
        if isinstance(val_a, (int, float, np.number)) and isinstance(val_b, (int, float, np.number)):
            try:
                stat_delta = float(val_a) - float(val_b)
            except Exception:
                stat_delta = None
        pct = float(impact / delta * 100) if delta != 0 else None
        reasons.append(
            {
                "feature": feat,
                "value_a": None if pd.isna(val_a) else val_a,
                "value_b": None if pd.isna(val_b) else val_b,
                "stat_delta": stat_delta,
                "impact": float(impact),
                "percent_of_delta": pct,
            }
        )
    return reasons


def _text_summary(player_a: str, player_b: str, delta: float, reasons: List[Dict[str, float]]) -> str:
    if not reasons:
        return f"{player_a} vs {player_b}: model predicts a {delta:.2f} OVR gap."
    direction = "higher" if delta >= 0 else "lower"
    leader = player_a if delta >= 0 else player_b
    tail = ", ".join([f"{r['feature']} ({r['impact']:+.2f})" for r in reasons[:3]])
    return f"{leader} is {abs(delta):.2f} OVR {direction} mainly due to {tail}."


def compare_players(
    player_a_id: int,
    player_b_id: int,
    model: Pipeline,
    config: TrainingConfig,
    raw_df: pd.DataFrame | None = None,
) -> PlayerComparison:
    """Produce a SHAP-backed comparison between two players."""
    if raw_df is None:
        raw_df = load_player_data(config.data_path, config.age_bins)
    raw_rows = {
        "a": raw_df[raw_df["ID"] == player_a_id],
        "b": raw_df[raw_df["ID"] == player_b_id],
    }
    for key, val in raw_rows.items():
        if val.empty:
            raise KeyError(f"Player ID not found: {player_a_id if key=='a' else player_b_id}")

    df = clean_dataframe(raw_df, config)
    row_a = df.loc[raw_rows["a"].index]
    row_b = df.loc[raw_rows["b"].index]
    feature_names = get_feature_names(model)

    pred_a = float(model.predict(row_a.drop(columns=[config.target], errors="ignore"))[0])
    pred_b = float(model.predict(row_b.drop(columns=[config.target], errors="ignore"))[0])
    delta = pred_a - pred_b

    preprocessor = model.named_steps["preprocessor"]
    model_only = model.named_steps["model"]

    Xa = preprocessor.transform(row_a.drop(columns=[config.target], errors="ignore"))
    Xb = preprocessor.transform(row_b.drop(columns=[config.target], errors="ignore"))
    if hasattr(Xa, "toarray"):
        Xa = Xa.toarray()
    if hasattr(Xb, "toarray"):
        Xb = Xb.toarray()
    Xa = np.asarray(Xa, dtype=float)
    Xb = np.asarray(Xb, dtype=float)

    explainer = shap.TreeExplainer(model_only)
    shap_a = explainer.shap_values(Xa)[0]
    shap_b = explainer.shap_values(Xb)[0]

    delta_contribs = shap_a - shap_b
    aggregated = _aggregate_deltas(feature_names, delta_contribs)
    reasons = _summarize_reasons(aggregated, row_a.squeeze(), row_b.squeeze(), delta)
    summary_text = _text_summary(raw_rows["a"]["Name"].iloc[0], raw_rows["b"]["Name"].iloc[0], delta, reasons)

    feature_table = pd.DataFrame(
        {
            "feature": aggregated.index,
            "contribution_delta": aggregated.values,
        }
    )

    # Simple what-if: adjust top positive driver for player B by +5 and recompute gap.
    what_if_results: List[Dict[str, float]] = []
    if reasons:
        # Pick the largest absolute impact feature.
        candidate = reasons[0]["feature"]
        # Clone row_b and adjust numeric feature if present.
        adjusted_b = row_b.copy()
        if candidate in adjusted_b and isinstance(adjusted_b[candidate], (int, float, np.number)):
            adjusted_b[candidate] = adjusted_b[candidate] + 5
        # Predict with adjusted B.
        pred_a = float(model.predict(row_a.drop(columns=[config.target], errors="ignore"))[0])
        pred_b_adj = float(model.predict(adjusted_b.drop(columns=[config.target], errors="ignore"))[0])
        delta_adj = pred_a - pred_b_adj
        what_if_results.append(
            {
                "feature": candidate,
                "adjustment": "+5",
                "new_delta": delta_adj,
                "description": f"If Player B's {candidate} increased by +5, predicted OVR gap would be {delta_adj:+.2f}.",
            }
        )

    return PlayerComparison(
        player_a_name=raw_rows["a"]["Name"].iloc[0],
        player_b_name=raw_rows["b"]["Name"].iloc[0],
        pred_a=pred_a,
        pred_b=pred_b,
        delta=delta,
        top_reasons=reasons,
        natural_language_summary=summary_text,
        feature_table=feature_table,
        what_if=what_if_results,
    )
