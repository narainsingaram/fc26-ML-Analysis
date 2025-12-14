#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def compute_causal_effects(df: pd.DataFrame, target: str = "OVR", top_k: int = 20) -> pd.DataFrame:
    """Approximate causal effects via partial regression controlling for confounders."""
    df = df.copy()
    numeric_cols = [
        c for c in df.select_dtypes(include=["int64", "float64"]).columns if c not in {target, "ID", "Rank"}
    ]
    # Confounders: league, team, position, age
    categorical_cols = [c for c in ["League", "Team", "Position"] if c in df.columns]
    if "Age" in df.columns and "Age" not in numeric_cols:
        numeric_cols.append("Age")

    X = df[numeric_cols + categorical_cols]
    y = df[target]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("enc", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ]
    )
    model = Ridge(alpha=1.0)
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X, y)

    coefs = model.coef_[: len(numeric_cols)]  # first block corresponds to numeric features
    effects = pd.DataFrame(
        {
            "feature": numeric_cols,
            "per_unit_effect": coefs,
            "per_five_effect": coefs * 5,
        }
    )
    effects["abs_effect"] = effects["per_five_effect"].abs()
    effects = effects.sort_values(by="abs_effect", ascending=False).head(top_k).reset_index(drop=True)
    return effects[["feature", "per_unit_effect", "per_five_effect"]]


def main():
    parser = argparse.ArgumentParser(description="Estimate causal-like effects of attributes on OVR via partial regression.")
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/EAFC26.csv"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/causal_effects.json"))
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    effects = compute_causal_effects(df)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    effects.to_json(args.output_path, orient="records", indent=2)
    print(f"Saved causal effects to {args.output_path}")
    print(effects.head(8))


if __name__ == "__main__":
    main()
