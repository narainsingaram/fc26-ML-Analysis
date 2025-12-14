from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _extract_number_from_text(series: pd.Series, pattern: str) -> pd.Series:
    """Extract the first numeric match using a regex pattern; returns float series."""
    return series.str.extract(pattern)[0].astype(float)


def add_physical_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse height/weight text fields into numeric columns."""
    result = df.copy()
    if "Height" in result.columns:
        result["height_cm"] = _extract_number_from_text(result["Height"], r"([0-9]+)cm")
    if "Weight" in result.columns:
        result["weight_kg"] = _extract_number_from_text(result["Weight"], r"([0-9]+)kg")
    return result


def add_age_buckets(df: pd.DataFrame, bins: Iterable[int]) -> pd.DataFrame:
    """Add an age bucket column to support bias slices."""
    result = df.copy()
    if "Age" in result.columns:
        result["age_bucket"] = pd.cut(result["Age"], bins=bins, include_lowest=True)
    return result


def load_player_data(path: Path, age_bins: Iterable[int]) -> pd.DataFrame:
    """Load raw EA FC26 data and enrich with helper columns."""
    df = pd.read_csv(path)
    df = add_physical_attributes(df)
    df = add_age_buckets(df, bins=age_bins)
    return df
