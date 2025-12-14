from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def group_error_analysis(
    y_true, y_pred, group: pd.Series, min_count: int = 25
) -> pd.DataFrame:
    """Compute error metrics by group and drop under-supported slices."""
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": group})
    grouped = frame.groupby("group").agg(
        count=("y_true", "size"),
        mae=("y_true", lambda y: np.mean(np.abs(y - frame.loc[y.index, "y_pred"]))),
        rmse=(
            "y_true",
            lambda y: np.sqrt(np.mean((y - frame.loc[y.index, "y_pred"]) ** 2)),
        ),
        mean_error=("y_true", lambda y: np.mean(frame.loc[y.index, "y_pred"] - y)),
    )
    grouped = grouped[grouped["count"] >= min_count]
    return grouped.sort_values(by="mae").reset_index()


def collect_bias_reports(
    df: pd.DataFrame,
    y_pred,
    groups: Dict[str, pd.Series],
    target: str = "OVR",
    min_count: int = 25,
) -> Dict[str, pd.DataFrame]:
    reports = {}
    y_true = df[target]
    for name, series in groups.items():
        reports[name] = group_error_analysis(y_true, y_pred, series, min_count=min_count)
    return reports


def save_bias_reports(reports: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        report.to_csv(output_dir / f"{name}_bias.csv", index=False)
