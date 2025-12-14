# FC26 Rating Engine

Reverse-engineer EA Sports FC26 player OVR ratings from raw attributes with a production-style Python codebase (no notebooks required).

## Project layout
- `data/raw/EAFC26.csv` – supplied player dataset (men + women)
- `src/rating_engine/` – reusable modules for data loading, feature prep, modeling, explainability, and bias analysis
- `scripts/train.py` – end-to-end training/evaluation entrypoint
- `artifacts/` – outputs (metrics, bias reports, feature importances, trained model)

## Quickstart
```
pip install -r requirements.txt
python scripts/train.py --data-path data/raw/EAFC26.csv --output-dir artifacts
```

## What the pipeline does
- Loads the FC26 CSV, parses height/weight, and builds numeric/categorical features
- Fits a RandomForest regressor with preprocessing (scaling + one-hot encoding)
- Evaluates MAE/RMSE/R² on a stratified test split
- Computes permutation feature importance for interpretability
- Runs bias/error slices across position, gender, league, and age buckets
- Saves `artifacts/metrics.json`, `feature_importance.csv`, `bias/*.csv`, and `model.joblib`

### Position-aware modeling
Train role-specific models (GK/DEF/MID/ATT) and summarize drivers:
```
python scripts/position_experiment.py
```
Outputs live in `artifacts/position_models/` and `experiments/position_comparison.md`.

### Player comparison (“Why is A rated higher than B?”)
Explain OVR deltas between two players with SHAP-backed counterfactuals using the trained model:
```
python scripts/compare_players.py --player-a-id 209331 --player-b-id 227203
```
Writes a JSON explanation to stdout and a markdown report to `experiments/player_comparison.md`.

### Lightweight web UI
Spin up a small Flask app that serves an interactive comparator:
```
python app.py
# open http://localhost:8000
```
Type to filter players, pick Player A/B, and the page will call the same backend `/api/compare` and show the SHAP-driven factors and OVR delta inline.

## Position-conditioned models
- Train separate models per positional macro-group (GK/DEF/MID/ATT) to avoid averaging across roles:
```
python scripts/train_position_models.py
```
Outputs live in `artifacts/position_ensemble/` with `ensemble.joblib` and a metrics report. The `PositionEnsembleModel` routes predictions to the right sub-model based on the player's position group.

## PyTorch tabular model (embeddings)
- Train a neural model with learned embeddings for categorical features + MLP for numerics:
```
python scripts/train_torch_model.py --epochs 8
```
Artifacts land in `artifacts/torch_model/` (`model.pth`, `meta.json`). The API will automatically use this model for what-if slider predictions if present, while the scikit pipeline remains for compare/SHAP.

## Causal signals
- Estimate causal-like effects (partial regression controlling for confounders: league, team, position) of attributes on OVR:
```
python scripts/causal_effects.py
```
Outputs `artifacts/causal_effects.json`, which the UI consumes to show top ΔOVR per +1/+5.

## Anomaly detector (overrated/underrated)
- Build residuals (Actual - Model) to surface anomalies:
```
PYTHONPATH=src python3 scripts/build_anomalies.py
```
Produces `artifacts/anomalies.json` and powers the UI's anomalies panel for filtering underrated/overrated players (by position/league/age).

## Quantile regression (P10/P50/P90)
- Train a TensorFlow quantile model to get floor/median/ceiling OVR estimates:
```
PYTHONPATH=src python3 scripts/train_quantile_model.py --epochs 8
```
Outputs to `artifacts/quantile_model/` (SavedModel + preprocessor). The UI will show P10/P50/P90 for the compared Player B if the quantile model is present. If TensorFlow is missing, install it first.


## Notes
- The code is framework-light (pandas + scikit-learn) so it can run anywhere without notebooks.
- Modify `TrainingConfig` in `src/rating_engine/config.py` to adjust split sizes, top leagues, or categorical columns.
