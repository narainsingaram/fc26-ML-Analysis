from __future__ import annotations

import ast
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import TrainingConfig

# Position group mapping for role-fit heuristics.
POSITION_GROUPS = {
    "GK": {"GK"},
    "DEF": {"CB", "RB", "LB", "RWB", "LWB", "RCB", "LCB"},
    "MID": {"CM", "CDM", "CAM", "LM", "RM", "LAM", "RAM", "LDM", "RDM"},
    "ATT": {"ST", "CF", "RW", "LW", "RF", "LF"},
}
GROUP_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "ATT": 3}


def parse_alt_positions(value) -> List[str]:
    """Return a normalized list of alternative positions from the raw text cell."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        vals = value
    elif isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            vals = parsed if isinstance(parsed, (list, tuple)) else [parsed]
        except Exception:
            vals = value.replace("[", "").replace("]", "").split(",")
    else:
        vals = []
    cleaned = []
    for v in vals:
        if v is None:
            continue
        text = str(v).strip().upper().replace("'", "").replace('"', "")
        if not text:
            continue
        cleaned.append(text)
    return cleaned


def map_position_group(position: str) -> str:
    """Map a detailed position to a coarse role group."""
    pos = (position or "").upper()
    for group, positions in POSITION_GROUPS.items():
        if pos in positions:
            return group
    return "OTHER"


def _role_group_distance(primary_pos: str, alt_positions: List[str]) -> float:
    """Compute distance between primary group and nearest alt group (0=same, 1=adjacent, etc.)."""
    primary_group = map_position_group(primary_pos)
    primary_code = GROUP_ORDER.get(primary_group)
    if primary_code is None:
        return 0.0
    alt_codes = [
        GROUP_ORDER.get(map_position_group(p))
        for p in alt_positions
        if GROUP_ORDER.get(map_position_group(p)) is not None
    ]
    if not alt_codes:
        return 0.0
    return float(min(abs(primary_code - c) for c in alt_codes))


def _simplify_leagues(df: pd.DataFrame, max_leagues: int) -> pd.Series:
    """Keep the most common leagues; group the tail under 'Other'."""
    league_counts = df["League"].value_counts()
    keep = set(league_counts.head(max_leagues).index)
    return df["League"].apply(lambda x: x if x in keep else "Other")


def clean_dataframe(df: pd.DataFrame, config: TrainingConfig, drop_null_only: bool = True) -> pd.DataFrame:
    """Drop unused columns and create a simplified league feature.

    drop_null_only: if True, remove columns that are entirely NaN. Set to False when predicting
    on single rows to preserve schema expected by trained models.
    """
    enriched = df.copy()

    if "Alternative positions" in enriched.columns and "Position" in enriched.columns:
        alt_lists = enriched["Alternative positions"].apply(parse_alt_positions)
        enriched["alt_position_count"] = alt_lists.apply(len).astype(int)
        enriched["has_alt_role"] = (enriched["alt_position_count"] > 0).astype(int)
        enriched["role_group_distance"] = [
            _role_group_distance(pos, alts) for pos, alts in zip(enriched["Position"], alt_lists)
        ]
    else:
        enriched["alt_position_count"] = 0
        enriched["has_alt_role"] = 0
        enriched["role_group_distance"] = 0.0

    drop_cols = set(config.drop_columns).union({"Height", "Weight"})
    available_drop = [c for c in drop_cols if c in df.columns]
    cleaned = enriched.drop(columns=available_drop, errors="ignore")

    if "League" in cleaned.columns:
        cleaned["League"] = _simplify_leagues(cleaned, config.top_leagues)
    if drop_null_only:
        # Remove columns that are entirely missing to avoid noisy warnings downstream.
        null_only = [c for c in cleaned.columns if cleaned[c].isna().all()]
        if null_only:
            cleaned = cleaned.drop(columns=null_only)
    return cleaned


def infer_feature_types(df: pd.DataFrame, config: TrainingConfig) -> Tuple[List[str], List[str]]:
    """Infer numeric and categorical columns after cleaning."""
    categorical_cols = [c for c in config.categorical_columns if c in df.columns]
    if "age_bucket" in df.columns and "age_bucket" not in categorical_cols:
        categorical_cols.append("age_bucket")

    numeric_cols = [
        c for c in df.select_dtypes(include=["int64", "float64"]).columns if c != config.target
    ]
    numeric_cols = [c for c in numeric_cols if c not in categorical_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(
    numeric_cols: List[str], categorical_cols: List[str]
) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols) if numeric_cols else ("num", "drop", numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
            if categorical_cols
            else ("cat", "drop", categorical_cols),
        ]
    )


def split_features_target(df: pd.DataFrame, target: str):
    """Return feature matrix and target vector."""
    if target not in df.columns:
        raise KeyError(f"Target column {target} missing from dataframe")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
