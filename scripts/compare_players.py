#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from rating_engine.compare import PlayerComparison, compare_players  # noqa: E402
from rating_engine.config import DEFAULT_CONFIG, TrainingConfig  # noqa: E402
from rating_engine.data import load_player_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain why Player A is rated higher than Player B.")
    parser.add_argument("--player-a-id", type=int, required=True, help="EA FC player ID for Player A.")
    parser.add_argument("--player-b-id", type=int, required=True, help="EA FC player ID for Player B.")
    parser.add_argument(
        "--model-path", type=Path, default=Path("artifacts/model.joblib"), help="Trained pipeline with preprocessor."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_CONFIG.data_path, help="Path to EA FC26 CSV.")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("experiments/player_comparison.md"),
        help="Where to write the markdown summary.",
    )
    parser.add_argument("--json-path", type=Path, help="Optional path to save JSON output.")
    return parser.parse_args()


def save_markdown(result: PlayerComparison, path: Path) -> None:
    lines = []
    lines.append("# Player Comparison")
    lines.append("")
    lines.append(f"**{result.player_a_name}** vs **{result.player_b_name}**")
    lines.append("")
    lines.append(f"- Predicted OVR: {result.pred_a:.2f} vs {result.pred_b:.2f} (Δ = {result.delta:+.2f})")
    lines.append(f"- Why: {result.natural_language_summary}")
    lines.append("")
    lines.append("## Top Factors Driving the Gap")
    for reason in result.top_reasons:
        impact = reason.get("impact", 0.0)
        pct = reason.get("percent_of_delta")
        pct_txt = f" ({pct:.0f}% of Δ)" if pct is not None else ""
        lines.append(
            f"- {reason['feature']}: A={reason.get('value_a','—')} / B={reason.get('value_b','—')} | "
            f"Stat Δ={reason.get('stat_delta','—')} | Impact={impact:+.3f}{pct_txt}"
        )
    lines.append("")
    lines.append("## Full Contribution Table")
    lines.append("```")
    lines.append(result.feature_table.to_string(index=False))
    lines.append("```")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    config = TrainingConfig(data_path=args.data_path)
    df_raw = load_player_data(config.data_path, config.age_bins)

    model = joblib.load(args.model_path)
    result = compare_players(
        player_a_id=args.player_a_id,
        player_b_id=args.player_b_id,
        model=model,
        config=config,
        raw_df=df_raw,
    )

    def cast_num(x):
        if x is None:
            return None
        try:
            if isinstance(x, (int, float, np.number)):
                return float(x)
            return x
        except Exception:
            return x

    top_reasons = []
    for r in result.top_reasons:
        top_reasons.append(
            {
                "feature": r["feature"],
                "value_a": cast_num(r.get("value_a")),
                "value_b": cast_num(r.get("value_b")),
                "stat_delta": cast_num(r.get("stat_delta")),
                "impact": cast_num(r.get("impact")),
                "percent_of_delta": cast_num(r.get("percent_of_delta")),
            }
        )

    payload = {
        "player_a": {"name": result.player_a_name, "predicted_ovr": float(result.pred_a)},
        "player_b": {"name": result.player_b_name, "predicted_ovr": float(result.pred_b)},
        "ovr_difference": float(result.delta),
        "top_reasons": top_reasons,
        "natural_language_summary": result.natural_language_summary,
    }
    print(json.dumps(payload, indent=2))

    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(payload, indent=2))

    save_markdown(result, args.report_path)
    print(f"Markdown report written to {args.report_path}")


if __name__ == "__main__":
    main()
