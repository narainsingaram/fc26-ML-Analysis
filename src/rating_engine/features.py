from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import TrainingConfig


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
    drop_cols = set(config.drop_columns).union({"Height", "Weight"})
    available_drop = [c for c in drop_cols if c in df.columns]
    cleaned = df.drop(columns=available_drop, errors="ignore").copy()

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
