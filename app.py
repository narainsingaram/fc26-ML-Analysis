#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
import hashlib
import io
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, request, send_from_directory, send_file
import joblib
import pandas as pd

# Ensure local src/ is on the path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from rating_engine.compare import compare_players  # noqa: E402
from rating_engine.config import DEFAULT_CONFIG, TrainingConfig  # noqa: E402
from rating_engine.data import load_player_data  # noqa: E402
from rating_engine.features import clean_dataframe, map_position_group, parse_alt_positions  # noqa: E402
from rating_engine.similarity import build_similarity_index  # noqa: E402
from rating_engine.torch_predict import load_torch_predictor  # noqa: E402
from rating_engine.quantile_tf import load_quantile_predictor  # noqa: E402

app = Flask(__name__, static_folder="frontend", static_url_path="")


@lru_cache()
def get_config() -> TrainingConfig:
    return DEFAULT_CONFIG


@lru_cache()
def get_data() -> pd.DataFrame:
    cfg = get_config()
    return load_player_data(cfg.data_path, cfg.age_bins)


@lru_cache()
def get_model():
    model_path = Path("artifacts/model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run `python scripts/train.py` first."
        )
    return joblib.load(model_path)


@lru_cache()
def get_similarity_index():
    df = get_data()
    cfg = get_config()
    return build_similarity_index(df, cfg)

@lru_cache()
def get_torch_predictor():
    model_dir = Path("artifacts/torch_model")
    if not (model_dir / "model.pth").exists():
        return None
    return load_torch_predictor(model_dir)

@lru_cache()
def get_quantile_predictor():
    model_dir = Path("artifacts/quantile_model")
    return load_quantile_predictor(model_dir)

@lru_cache()
def get_uncertainty_predictor():
    from rating_engine.uncertainty import load_uncertainty_predictor
    model_dir = Path("artifacts/quantile_model")
    return load_uncertainty_predictor(model_dir)

@lru_cache()
def get_causal_effects():
    path = Path("artifacts/causal_effects.json")
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []

@lru_cache()
def get_anomalies():
    path = Path("artifacts/anomalies.json")
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


@lru_cache()
def get_model_meta():
    meta_path = Path("artifacts/model_meta.json")
    leaderboard_path = Path("artifacts/model_leaderboard.json")
    payload = {}
    if meta_path.exists():
        try:
            payload.update(json.loads(meta_path.read_text()))
        except Exception:
            payload["error"] = "could not read model_meta.json"
    if leaderboard_path.exists():
        try:
            payload["leaderboard"] = json.loads(leaderboard_path.read_text()).get("leaderboard", [])
        except Exception:
            payload.setdefault("leaderboard_error", "could not read model_leaderboard.json")
    return payload


def get_model_hash() -> str:
    model_path = Path("artifacts/model.joblib")
    if not model_path.exists():
        return "unavailable"
    h = hashlib.sha256()
    with model_path.open("rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:12]


@lru_cache()
def get_pred_cache():
    """Cache model predictions for all players for fast formation analysis and suggestions."""
    cfg = get_config()
    raw = get_data()
    model = get_model()
    df_clean = clean_dataframe(raw, cfg)
    X = df_clean.drop(columns=[cfg.target], errors="ignore")
    preds = model.predict(X)
    id_series = raw.loc[df_clean.index, "ID"]
    cache = {}
    for pid, pred in zip(id_series, preds):
        cache[int(pid)] = float(pred)
    return cache


def build_versatility_lookup(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict[int, dict]:
    """Return per-player versatility metrics aligned with cleaned features."""
    lookup: dict[int, dict] = {}
    for idx, row in cleaned_df.iterrows():
        raw_row = raw_df.loc[idx] if idx in raw_df.index else None
        if raw_row is None or raw_row is pd.NA:
            continue
        try:
            player_id = int(raw_row.get("ID"))
        except Exception:
            continue
        alt_positions = parse_alt_positions(raw_row.get("Alternative positions"))
        alt_groups = sorted({map_position_group(p) for p in alt_positions}) if alt_positions else []
        lookup[player_id] = {
            "alt_positions": alt_positions,
            "alt_position_count": int(row["alt_position_count"]) if "alt_position_count" in row else len(alt_positions),
            "has_alt_role": bool(row["has_alt_role"]) if "has_alt_role" in row else bool(alt_positions),
            "role_group_distance": float(row["role_group_distance"]) if "role_group_distance" in row else 0.0,
            "primary_group": map_position_group(raw_row.get("Position")),
            "alt_groups": alt_groups,
        }
    return lookup


def forecast_player(player_id: int):
    cfg = get_config()
    df = get_data()
    model = get_model()
    torch_predictor = get_torch_predictor()

    row = df[df["ID"] == player_id]
    if row.empty:
        raise KeyError(f"Player ID {player_id} not found")
    cleaned = clean_dataframe(row, cfg, drop_null_only=False)
    X = cleaned.drop(columns=[cfg.target], errors="ignore")
    base_pred = float(torch_predictor.predict_df(X)[0]) if torch_predictor else float(model.predict(X)[0])

    age = float(row["Age"].iloc[0]) if "Age" in row else 26.0
    residual = base_pred - float(row[cfg.target].iloc[0])
    age_factor = max(0.0, min(1.0, (23.0 - age) / 10.0))
    growth = age_factor * 2.0 + 0.3 * residual
    future_ovr = base_pred + growth

    # simple prob via sigmoid on growth and residual
    score = age_factor * 2.0 + residual * 0.8
    upgrade_prob = 1 / (1 + (2.71828 ** (-score)))

    drivers = [
        {"feature": "Age", "reason": f"Young age boosts growth factor ({age_factor:.2f})."},
        {"feature": "Residual", "reason": f"Model residual {residual:+.2f} indicates headroom."},
    ]
    return {
        "current_pred": base_pred,
        "future_pred": future_ovr,
        "delta": future_ovr - base_pred,
        "upgrade_prob": upgrade_prob,
        "drivers": drivers,
    }


def predict_with_adjustments(player_id: int, adjustments: dict) -> dict:
    """Predict OVR for a player after applying attribute adjustments."""
    cfg = get_config()
    df = get_data()
    model = get_model()
    torch_predictor = get_torch_predictor()
    row = df[df["ID"] == player_id]
    if row.empty:
        raise KeyError(f"Player ID {player_id} not found")

    cleaned = clean_dataframe(row, cfg, drop_null_only=False)
    X_base = cleaned.drop(columns=[cfg.target], errors="ignore")
    if torch_predictor:
        base_pred = float(torch_predictor.predict_df(X_base)[0])
    else:
        base_pred = float(model.predict(X_base)[0])

    adjusted = cleaned.copy()
    for col, val in adjustments.items():
        if col in adjusted.columns:
            adjusted[col] = val
    X_adj = adjusted.drop(columns=[cfg.target], errors="ignore")
    if torch_predictor:
        adj_pred = float(torch_predictor.predict_df(X_adj)[0])
    # Calculate partial dependence / delta
    delta = adj_pred - base_pred
    
    # Calibration: Anchor to Actual OVR
    actual_ovr = row.iloc[0].get("OVR")
    if actual_ovr is not None and not pd.isna(actual_ovr):
        actual_val = float(actual_ovr)
        residual = actual_val - base_pred
        # Apply calibration
        final_baseline = actual_val
        final_adjusted = adj_pred + residual
    else:
        # Fallback if no actual OVR (unlikely for existing players)
        final_baseline = base_pred
        final_adjusted = adj_pred

    unc_predictor = get_uncertainty_predictor() 
    base_u = None
    adj_u = None
    if unc_predictor:
        try:
            base_u = unc_predictor.predict(X_base)[0]
            adj_u = unc_predictor.predict(X_adj)[0]
        except Exception:
            pass

    def fmt_u(u):
        if not u: return None
        return {
            "p10": u.p10, "p90": u.p90, "width": u.interval_width,
            "is_extrapolating": u.is_extrapolating,
            "warnings": u.warnings
        }

    return {
        "baseline": final_baseline,
        "adjusted": final_adjusted,
        "delta": final_adjusted - final_baseline,
        "raw_baseline": base_pred, # useful for debugging
        "baseline_uncertainty": fmt_u(base_u),
        "adjusted_uncertainty": fmt_u(adj_u),
    }


@app.route("/api/players")
def list_players():
    try:
        # 1. Parsing Params
        limit = int(request.args.get("limit", 100))
        search = request.args.get("search", "").lower().strip()
        sort_by = request.args.get("sort", "ovr_desc")  # ovr_desc, age_asc, etc.
        
        # Ranges
        min_ovr = int(request.args.get("min_ovr", 0))
        max_ovr = int(request.args.get("max_ovr", 99))
        min_age = int(request.args.get("min_age", 15))
        max_age = int(request.args.get("max_age", 50))
        min_height = int(request.args.get("min_height", 0))
        max_height = int(request.args.get("max_height", 250))
        min_weight = int(request.args.get("min_weight", 0))
        max_weight = int(request.args.get("max_weight", 150))
        
        # Skills
        min_wf = int(request.args.get("min_wf", 1))
        min_sm = int(request.args.get("min_sm", 1))
        
        # Categorical
        pos = request.args.get("position", "").strip()
        league = request.args.get("league", "").strip()
        nation = request.args.get("nation", "").strip()
        playstyle = request.args.get("playstyle", "").lower().strip()
        
        df = get_data()
        
        # 2. Apply Filters
        mask = pd.Series(True, index=df.index)
        
        # OVR
        mask &= (df["OVR"] >= min_ovr) & (df["OVR"] <= max_ovr)
        
        # Search
        if search:
            mask &= df["Name"].str.lower().str.contains(search)
            
        # Position (exact match or comma-list)
        if pos and pos.lower() != "any":
            # Handle multi-select like "ST,CF"
            pos_list = [p.strip() for p in pos.split(",")]
            mask &= df["Position"].isin(pos_list)
            
        # League / Nation
        if league:
            mask &= (df["League"] == league)
        if nation:
            mask &= (df["Nation"] == nation)
            
        # Physical
        if min_age > 15 or max_age < 50:
            mask &= (df["Age"] >= min_age) & (df["Age"] <= max_age)
        if min_height > 0 or max_height < 250:
            # use height_cm
            mask &= (df["height_cm"] >= min_height) & (df["height_cm"] <= max_height)
        if min_weight > 0 or max_weight < 150:
            # use weight_kg
            mask &= (df["weight_kg"] >= min_weight) & (df["weight_kg"] <= max_weight)
            
        # Skills
        if min_wf > 1:
            mask &= (df["Weak foot"] >= min_wf)
        if min_sm > 1:
            mask &= (df["Skill moves"] >= min_sm)
            
        # PlayStyle (text match)
        if playstyle:
            # 'play style' column
            mask &= df["play style"].str.lower().str.contains(playstyle, na=False)

        subset = df[mask].copy()
        
        # 3. Sorting
        if sort_by == "ovr_desc":
            subset = subset.sort_values(by="OVR", ascending=False)
        elif sort_by == "ovr_asc":
            subset = subset.sort_values(by="OVR", ascending=True)
        elif sort_by == "age_asc":
            subset = subset.sort_values(by="Age", ascending=True)
        elif sort_by == "pot_desc": # potential proxy
            # Reuse logic? For now just sort OVR
            subset = subset.sort_values(by="OVR", ascending=False)
            
        subset = subset.head(limit)
        
        # 4. Response
        # Return expanded fields for display
        cols = [
            "ID", "Name", "Position", "OVR", "Team", "League", "Nation", "GENDER", 
            "Age", "height_cm", "weight_kg", "Weak foot", "Skill moves", "play style", "card"
        ]
        # select only existing cols
        final_cols = [c for c in cols if c in subset.columns]
        players = subset[final_cols].to_dict(orient="records")
        return jsonify({"players": players, "count": len(players)})

    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 500

@app.route("/api/metrics")
def metrics():
    """Lightweight metrics + distribution snapshot for the analytics dashboard."""
    try:
        raw = get_data()
        cfg = get_config()
        model = get_model()
        df = clean_dataframe(raw, cfg)
        if df.empty:
            return jsonify({"error": "No data available"}), 500
        y_true = df[cfg.target].astype(float)
        X = df.drop(columns=[cfg.target], errors="ignore")
        y_pred = model.predict(X)
        y_pred = np.asarray(y_pred, dtype=float)
        residuals = y_true.values - y_pred

        def safe_rmse(y, p):
            return float(np.sqrt(np.mean((y - p) ** 2)))

        def safe_mae(y, p):
            return float(np.mean(np.abs(y - p)))

        overall = {
            "rmse": safe_rmse(y_true, y_pred),
            "mae": safe_mae(y_true, y_pred),
            "r2": float(1 - np.sum((y_true - y_pred) ** 2) / (np.sum((y_true - np.mean(y_true)) ** 2) + 1e-9)),
        }

        # align metadata
        raw_aligned = raw.loc[df.index]
        pred_df = raw_aligned.assign(pred=y_pred, actual=y_true.values)

        def rmse_group(grp):
            return safe_rmse(grp["actual"].values, grp["pred"].values)

        by_pos = (
            pred_df.groupby("Position")
            .apply(lambda g: pd.Series({"rmse": rmse_group(g), "count": len(g)}))
            .reset_index()
            .sort_values("rmse")
        )

        by_gender = (
            pred_df.groupby("GENDER")
            .apply(lambda g: pd.Series({"rmse": rmse_group(g), "count": len(g)}))
            .reset_index()
        )

        counts, bins = np.histogram(y_true, bins=15)

        # scatter sample
        sample = raw_aligned.assign(pred=y_pred, actual=y_true.values)
        sample = sample.reset_index(drop=True).head(800)
        scatter = [
          {
            "name": row["Name"],
            "pred": float(row["pred"]),
            "actual": float(row["actual"]),
            "position": row.get("Position"),
            "league": row.get("League"),
          }
          for _, row in sample.iterrows()
        ]

        # residual extremes
        res_df = raw_aligned.assign(pred=y_pred, actual=y_true.values, residual=residuals)
        underrated = (
            res_df.sort_values("residual", ascending=False)
            .head(20)[["Name", "Position", "League", "residual", "actual", "pred"]]
            .to_dict(orient="records")
        )
        overrated = (
            res_df.sort_values("residual", ascending=True)
            .head(20)[["Name", "Position", "League", "residual", "actual", "pred"]]
            .to_dict(orient="records")
        )

        return jsonify(
            {
                "overall": overall,
                "by_position": by_pos.to_dict(orient="records"),
                "by_gender": by_gender.to_dict(orient="records"),
                "hist": {"counts": counts.tolist(), "bins": bins.tolist()},
                "scatter": scatter,
                "underrated": underrated,
                "overrated": overrated,
                "model_meta": get_model_meta(),
            }
        )
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 500


@app.route("/api/model_meta")
def model_meta():
    data = get_model_meta()
    if not data:
        return jsonify({"error": "Model metadata not available"}), 404
    return jsonify(data)


@app.route("/api/feature_importance")
def feature_importance():
    """Return permutation importances saved during training."""
    path = Path("artifacts/feature_importance.csv")
    if not path.exists():
        return jsonify({"error": "feature_importance.csv not found; run training first."}), 404
    try:
        import pandas as pd
    except ImportError:
        return jsonify({"error": "pandas not available"}), 500
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 500
    limit = int(request.args.get("limit", 30))
    df = df.head(limit)
    return jsonify({"features": df.to_dict(orient="records")})


def _fit_penalty(player_pos: str, target_pos: str) -> tuple[float, str]:
    """Penalty (negative) for playing out of position plus label."""
    if not player_pos or not target_pos:
        return 0.0, "Unknown fit"
    # Normalize target to base pos (strip trailing digits for duplicate slots)
    base_target = str(target_pos).rstrip("0123456789")
    if player_pos == base_target:
        return 0.0, "Natural fit"
    player_group = map_position_group(player_pos)
    target_group = map_position_group(base_target)
    if player_group == target_group:
        return -1.0, "Same line"
    return -3.0, "Off line"


def _predict_for_ids(ids):
    cache = get_pred_cache()
    return {pid: cache.get(pid) for pid in ids}


def _suggest_for_position(pos: str, top_k: int = 5):
    base_pos = str(pos).rstrip("0123456789")
    raw = get_data()
    preds = get_pred_cache()
    df = raw.copy()
    df["predicted"] = df["ID"].apply(lambda x: preds.get(int(x), None))
    # Prioritize exact position, then same group.
    exact = df[df["Position"] == base_pos]
    if len(exact) < top_k:
        group = map_position_group(base_pos)
        extra = df[df["Position"].apply(map_position_group) == group]
        df = pd.concat([exact, extra]).drop_duplicates(subset=["ID"])
    else:
        df = exact
    df = df.dropna(subset=["predicted"])
    df = df.sort_values(by="predicted", ascending=False).head(top_k)
    suggestions = []
    for _, row in df.iterrows():
        penalty, fit = _fit_penalty(row["Position"], pos)
        suggestions.append(
            {
                "id": int(row["ID"]),
                "name": row.get("Name"),
                "position": row.get("Position"),
                "league": row.get("League"),
                "team": row.get("Team"),
                "predicted_ovr": float(row["predicted"]),
                "adjusted_ovr": float(row["predicted"] + penalty),
                "fit": fit,
            }
        )
    return suggestions


def _chemistry_score(player_ids):
    try:
        index = get_similarity_index()
    except Exception:
        return None, None
    sims = []
    weakest = None
    weakest_mean = None
    for pid in player_ids:
        try:
            matches = index.query(pid, top_k=len(player_ids))
            # Filter to those in the team
            filtered = [m for m in matches if m["id"] in player_ids and m["id"] != pid]
            vals = [m["similarity"] for m in filtered]
            if vals:
                mean_sim = float(np.mean(vals))
                sims.extend(vals)
                if weakest_mean is None or mean_sim < weakest_mean:
                    weakest_mean = mean_sim
                    weakest = pid
        except Exception:
            continue
    if not sims:
        return None, None
    return float(np.mean(sims) * 100), weakest


@app.route("/api/formation/analyze", methods=["POST"])
def formation_analyze():
    payload = request.get_json(silent=True) or {}
    formation = payload.get("formation") or []
    assignments = payload.get("assignments") or {}
    if not formation or not isinstance(formation, list):
        return jsonify({"error": "formation must be a list of positions"}), 400

    raw = get_data()
    preds = get_pred_cache()
    results = []
    filled_preds = []
    assigned_ids = []

    for pos in formation:
        slot = {"position": pos}
        pid = assignments.get(pos)
        if pid is not None:
            try:
                pid = int(pid)
            except Exception:
                return jsonify({"error": f"Invalid player id for {pos}"}), 400
            row = raw[raw["ID"] == pid]
            if row.empty:
                return jsonify({"error": f"Player {pid} not found"}), 404
            row = row.iloc[0]
            base_pred = preds.get(pid)
            if base_pred is None:
                base_pred = float(row.get("OVR", 0))
            penalty, fit_label = _fit_penalty(row.get("Position"), pos)
            adj = base_pred + penalty
            slot.update(
                {
                    "player_id": pid,
                    "name": row.get("Name"),
                    "position": row.get("Position"),
                    "team": row.get("Team"),
                    "league": row.get("League"),
                    "predicted_ovr": base_pred,
                    "adjusted_ovr": adj,
                    "fit": fit_label,
                    "penalty": penalty,
                }
            )
            filled_preds.append(adj)
            assigned_ids.append(pid)
        results.append(slot)

    suggestions = {pos: _suggest_for_position(pos) for pos in formation}

    team_strength = float(np.mean(filled_preds)) if filled_preds else None
    chem_score, weakest = _chemistry_score(assigned_ids)
    chem_suggestion = None
    if weakest and assigned_ids:
        # suggest a replacement similar to weakest but not in team
        try:
            idx = get_similarity_index()
            sim_results = idx.query(weakest, top_k=8)
            replacement = next((r for r in sim_results if r["id"] not in assigned_ids), None)
            if replacement:
                chem_suggestion = {
                    "replace_player_id": weakest,
                    "suggested_id": replacement["id"],
                    "suggested_name": replacement["meta"].get("Name"),
                    "similarity": replacement["similarity"],
                }
        except Exception:
            pass

    return jsonify(
        {
            "slots": results,
            "team_strength": team_strength,
            "chemistry_score": chem_score,
            "chemistry_suggestion": chem_suggestion,
            "suggestions": suggestions,
        }
    )


def _load_bias_snapshots(max_rows: int = 3):
    bias_dir = Path("artifacts/bias")
    snapshots = {}
    if not bias_dir.exists():
        return snapshots
    for csv_path in bias_dir.glob("*_bias.csv"):
        try:
            df = pd.read_csv(csv_path).head(max_rows)
            snapshots[csv_path.stem] = df.to_dict(orient="records")
        except Exception:
            continue
    return snapshots


def _draw_bar_chart(c, x, y, width, label, value, max_val, color):
    bar_width = (value / max_val) * width if max_val else 0
    c.setFillColor(color)
    c.rect(x, y, bar_width, 12, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica", 8)
    c.drawString(x, y + 2, f"{label}")
    c.drawRightString(x + width, y + 2, f"{value:.3f}")


@app.route("/api/report/comparison")
def report_comparison():
    try:
        player_a_id = int(request.args.get("player_a_id", ""))
        player_b_id = int(request.args.get("player_b_id", ""))
    except ValueError:
        return jsonify({"error": "player_a_id and player_b_id must be integers"}), 400

    try:
        result = compare_players(
            player_a_id=player_a_id,
            player_b_id=player_b_id,
            model=get_model(),
            config=get_config(),
            raw_df=get_data(),
        )
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 500

    raw_df = get_data()
    row_a = raw_df[raw_df["ID"] == player_a_id].iloc[0]
    row_b = raw_df[raw_df["ID"] == player_b_id].iloc[0]
    meta = get_model_meta()
    bias_snaps = _load_bias_snapshots()

    buffer = io.BytesIO()
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.colors import Color
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 40, "FC26 Model Report: Player Comparison")
    c.setFont("Helvetica", 9)
    c.drawString(40, height - 56, f"Generated: {datetime.utcnow().isoformat()}Z")
    c.drawString(300, height - 56, f"Model: {meta.get('best_model', 'unknown')}  |  Hash: {get_model_hash()}")

    # Player overview
    y = height - 90
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"Anchor: {result.player_a_name} (Pred {result.pred_a:.2f}, OVR {row_a.get('OVR', '—')})")
    c.drawString(40, y - 16, f"Challenger: {result.player_b_name} (Pred {result.pred_b:.2f}, OVR {row_b.get('OVR', '—')})")
    c.drawString(40, y - 32, f"Delta: {result.delta:+.2f}")

    # Top drivers chart
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y - 56, "Drivers (SHAP)")
    top_reasons = result.top_reasons[:8]
    max_abs = max((abs(r["impact"]) for r in top_reasons), default=1)
    bar_y = y - 72
    for r in top_reasons:
        color = Color(0.13, 0.83, 0.73) if r["impact"] >= 0 else Color(0.96, 0.35, 0.51)
        _draw_bar_chart(c, 40, bar_y, 240, r["feature"], r["impact"], max_abs, color)
        bar_y -= 16

    # What-if
    c.setFont("Helvetica-Bold", 11)
    c.drawString(320, y - 56, "What-If Scenario")
    c.setFont("Helvetica", 9)
    if result.what_if:
        text = result.what_if[0]["description"]
        c.drawString(320, y - 72, text[:80])
        c.drawString(320, y - 86, text[80:160])
    else:
        c.drawString(320, y - 72, "Not available.")

    # Bias snapshot
    c.setFont("Helvetica-Bold", 11)
    c.drawString(320, y - 110, "Bias Snapshots")
    bias_y = y - 126
    if bias_snaps:
        for name, rows in list(bias_snaps.items())[:3]:
            c.setFont("Helvetica-Bold", 9)
            c.drawString(320, bias_y, name.replace("_bias", "").title())
            bias_y -= 12
            c.setFont("Helvetica", 8)
            for row in rows:
                c.drawString(320, bias_y, f"{row.get('group')}: MAE {row.get('mae'):.3f} | Count {row.get('count')}")
                bias_y -= 10
            bias_y -= 6
    else:
        c.setFont("Helvetica", 9)
        c.drawString(320, bias_y, "No bias reports found.")

    c.showPage()
    c.save()
    buffer.seek(0)
    filename = f"report_{player_a_id}_vs_{player_b_id}.pdf"
    return send_file(buffer, as_attachment=True, download_name=filename, mimetype="application/pdf")


@app.route("/api/predict")
def predict_players():
    try:
        ids_param = request.args.get("ids", "")
        if not ids_param:
            return jsonify({"error": "ids is required"}), 400
        id_list = [int(x) for x in ids_param.split(",") if x.strip()]
        if not id_list:
            return jsonify({"error": "No valid ids provided"}), 400
        if len(id_list) > 4:
            return jsonify({"error": "Maximum 4 players at once"}), 400
    except ValueError:
        return jsonify({"error": "ids must be integers"}), 400

    raw = get_data()
    cfg = get_config()
    model = get_model()
    raw_subset = raw[raw["ID"].isin(id_list)]
    if raw_subset.empty:
        return jsonify({"error": "No players found for ids"}), 404

    rows = clean_dataframe(raw_subset, cfg, drop_null_only=False)
    preds = model.predict(rows.drop(columns=[cfg.target], errors="ignore"))
    vers_lookup = build_versatility_lookup(raw_subset, rows)
    
    # Uncertainty quantification
    unc_predictor = get_uncertainty_predictor()
    unc_results = []
    if unc_predictor:
        try:
            X_unc = rows.drop(columns=[cfg.target], errors="ignore")
            unc_results = unc_predictor.predict(X_unc)
        except Exception:
            unc_results = [None] * len(rows)
    else:
        unc_results = [None] * len(rows)

    results: list[dict[str, float | str | int | dict]] = []
    for idx, (_, row) in enumerate(rows.iterrows()):
        raw_row = raw[raw["ID"] == row["ID"]].iloc[0]
        item = {
                "id": int(row["ID"]),
                "name": raw_row.get("Name"),
                "predicted_ovr": float(preds[idx]),
                "ovr": raw_row.get("OVR"),
                "position": raw_row.get("Position"),
                "team": raw_row.get("Team"),
                "league": raw_row.get("League"),
                "card": raw_row.get("card"),
            }
        item["versatility"] = vers_lookup.get(int(row["ID"]))
        
        u_res = unc_results[idx]
        if u_res:
            item["uncertainty"] = {
                "p10": u_res.p10,
                "p50": u_res.p50,
                "p90": u_res.p90,
                "width": u_res.interval_width,
                "distance": u_res.distance_to_train,
                "is_extrapolating": u_res.is_extrapolating,
                "is_high_uncertainty": u_res.is_high_uncertainty,
                "warnings": u_res.warnings,
            }
            # Adjust skepticism flag
            if u_res.warnings:
                item["skepticism_flag"] = True
                item["skepticism_reasons"] = u_res.warnings
            else:
                 item["skepticism_flag"] = False
        
        results.append(item)

    return jsonify({"players": results, "count": len(results)})


@app.route("/api/compare")
def compare():
    try:
        player_a_id = int(request.args.get("player_a_id", ""))
        player_b_id = int(request.args.get("player_b_id", ""))
    except ValueError:
        return jsonify({"error": "player_a_id and player_b_id must be integers"}), 400

    try:
        result = compare_players(
            player_a_id=player_a_id,
            player_b_id=player_b_id,
            model=get_model(),
            config=get_config(),
            raw_df=get_data(),
        )
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 500

    raw_df = get_data()
    cfg = get_config()
    row_a = raw_df[raw_df["ID"] == player_a_id].iloc[0]
    row_b = raw_df[raw_df["ID"] == player_b_id].iloc[0]
    subset_raw = raw_df[raw_df["ID"].isin([player_a_id, player_b_id])]
    cleaned_subset = clean_dataframe(subset_raw, cfg, drop_null_only=False)
    vers_lookup = build_versatility_lookup(subset_raw, cleaned_subset)

    def cast_num(x):
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return x

    payload = {
        "player_a": {
            "id": player_a_id,
            "name": result.player_a_name,
            "predicted_ovr": float(result.pred_a),
            "position": row_a.get("Position"),
            "team": row_a.get("Team"),
            "league": row_a.get("League"),
            "card": row_a.get("card"),
            "ovr": cast_num(row_a.get("OVR")),
            "versatility": vers_lookup.get(player_a_id),
        },
        "player_b": {
            "id": player_b_id,
            "name": result.player_b_name,
            "predicted_ovr": float(result.pred_b),
            "position": row_b.get("Position"),
            "team": row_b.get("Team"),
            "league": row_b.get("League"),
            "card": row_b.get("card"),
            "ovr": cast_num(row_b.get("OVR")),
            "versatility": vers_lookup.get(player_b_id),
        },
        "ovr_difference": float(result.delta),
        "top_reasons": [
            {
                "feature": r["feature"],
                "value_a": cast_num(r.get("value_a")),
                "value_b": cast_num(r.get("value_b")),
                "stat_delta": cast_num(r.get("stat_delta")),
                "impact": cast_num(r.get("impact")),
                "percent_of_delta": cast_num(r.get("percent_of_delta")),
            }
            for r in result.top_reasons
        ],
        "what_if": result.what_if,
        "natural_language_summary": result.natural_language_summary,
    }
    return jsonify(payload)


@app.route("/api/similar")
def similar_players():
    try:
        player_id = int(request.args.get("player_id", ""))
    except ValueError:
        return jsonify({"error": "player_id must be an integer"}), 400
    top_k = int(request.args.get("k", 10))
    try:
        index = get_similarity_index()
        results = index.query(player_id, top_k=top_k)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 500

    # Flatten meta for frontend convenience.
    enriched = []
    for r in results:
        meta = r["meta"]
        enriched.append(
            {
                "id": r["id"],
                "similarity": r["similarity"],
                "name": meta.get("Name"),
                "position": meta.get("Position"),
                "ovr": meta.get("OVR"),
                "team": meta.get("Team"),
                "league": meta.get("League"),
                "card": meta.get("card"),
            }
        )
    return jsonify({"player_id": player_id, "results": enriched})


@app.route("/api/whatif", methods=["POST", "GET"])
def what_if():
    if request.method == "GET":
        return jsonify({"error": "Use POST with player_id and adjustments"}), 405
    payload = request.get_json(silent=True) or {}
    try:
        player_id = int(payload.get("player_id", ""))
    except (ValueError, TypeError):
        return jsonify({"error": "player_id must be an integer"}), 400
    adjustments = payload.get("adjustments", {})
    if not isinstance(adjustments, dict):
        return jsonify({"error": "adjustments must be a dict of feature->value"}), 400
    try:
        result = predict_with_adjustments(player_id, adjustments)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@app.route("/api/causal")
def causal():
    effects = get_causal_effects()
    return jsonify({"effects": effects})


@app.route("/api/anomalies")
def anomalies():
    data = get_anomalies()
    if not data:
        return jsonify({"error": "anomalies not built; run scripts/build_anomalies.py"}), 404
    kind = request.args.get("kind", "underrated")
    position = request.args.get("position")
    league = request.args.get("league")
    max_age = request.args.get("max_age")
    limit = int(request.args.get("limit", 20))

    df = pd.DataFrame(data)
    if position:
        df = df[df["position"] == position]
    if league:
        df = df[df["league"] == league]
    if max_age:
        try:
            age_val = int(max_age)
            df = df[df["age"] <= age_val]
        except ValueError:
            pass

    if kind == "underrated":
        df = df.sort_values(by="residual", ascending=False)
    elif kind == "overrated":
        df = df.sort_values(by="residual", ascending=True)
    else:
        df = df.sort_values(by="residual", ascending=False)

    df = df.head(limit)
    return jsonify({"results": df.to_dict(orient="records")})


@app.route("/api/quantiles")
def quantiles():
    predictor = get_quantile_predictor()
    if predictor is None:
        return jsonify({"error": "quantile model not available; train with scripts/train_quantile_model.py"}), 404
    try:
        player_id = int(request.args.get("player_id", ""))
    except (ValueError, TypeError):
        return jsonify({"error": "player_id must be an integer"}), 400
    df = get_data()
    row = df[df["ID"] == player_id]
    if row.empty:
        return jsonify({"error": "player not found"}), 404
    cfg = get_config()
    try:
        cleaned = clean_dataframe(row, cfg, drop_null_only=False)
        X = cleaned.drop(columns=[cfg.target], errors="ignore")
        preds = predictor.predict(X)[0]
        return jsonify({"p10": float(preds[0]), "p50": float(preds[1]), "p90": float(preds[2])})
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": f"quantile inference failed: {exc}"}), 500




@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/analytics")
def serve_analytics():
    return send_from_directory(app.static_folder, "analytics.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
