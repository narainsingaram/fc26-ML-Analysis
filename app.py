#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
import joblib
import pandas as pd

# Ensure local src/ is on the path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from rating_engine.compare import compare_players  # noqa: E402
from rating_engine.config import DEFAULT_CONFIG, TrainingConfig  # noqa: E402
from rating_engine.data import load_player_data  # noqa: E402
from rating_engine.features import clean_dataframe  # noqa: E402
from rating_engine.similarity import build_similarity_index  # noqa: E402
from rating_engine.torch_predict import load_torch_predictor  # noqa: E402
from rating_engine.quantile_tf import load_quantile_predictor  # noqa: E402
import json
import json

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
    else:
        adj_pred = float(model.predict(X_adj)[0])

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
        "baseline": base_pred,
        "adjusted": adj_pred,
        "delta": adj_pred - base_pred,
        "baseline_uncertainty": fmt_u(base_u),
        "adjusted_uncertainty": fmt_u(adj_u),
    }


@app.route("/api/players")
def list_players():
    try:
        limit = int(request.args.get("limit", 400))
        search = request.args.get("search", "").lower().strip()
        df = get_data()
        cols = ["ID", "Name", "Position", "OVR", "Team", "League", "GENDER", "card"]
        subset = df[cols].copy()
        if search:
            subset = subset[subset["Name"].str.lower().str.contains(search)]
        subset = subset.sort_values(by="OVR", ascending=False).head(limit)
        players = subset.to_dict(orient="records")
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
            }
        )
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 500


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
    df_clean = clean_dataframe(raw, cfg)
    rows = df_clean[df_clean["ID"].isin(id_list)]
    if rows.empty:
        return jsonify({"error": "No players found for ids"}), 404

    preds = model.predict(rows.drop(columns=[cfg.target], errors="ignore"))
    
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
    row_a = raw_df[raw_df["ID"] == player_a_id].iloc[0]
    row_b = raw_df[raw_df["ID"] == player_b_id].iloc[0]

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
