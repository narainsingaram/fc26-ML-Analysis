from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

from .config import TrainingConfig
from .evaluation import regression_metrics
from .features import build_preprocessor, clean_dataframe, infer_feature_types, split_features_target

try:
    import optuna  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    optuna = None


@dataclass
class SearchResult:
    name: str
    pipeline: Pipeline
    holdout_metrics: Dict[str, float]
    params: Dict[str, float | int | str | None]


def _compute_monotone_constraints(preprocessor, target_feature: str = "Age") -> List[int]:
    """Generate a monotonic vector with +1 for the target_feature, 0 otherwise."""
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        return []
    constraints: List[int] = []
    for name in names:
        base = name.split("__")[-1]
        constraints.append(1 if base == target_feature else 0)
    return constraints


def _score_model(pipeline: Pipeline, X_test, y_test) -> Dict[str, float]:
    preds = pipeline.predict(X_test)
    metrics = regression_metrics(y_test, preds)
    metrics["mae"] = float(metrics["mae"])
    metrics["rmse"] = float(metrics["rmse"])
    metrics["r2"] = float(metrics["r2"])
    return metrics


def _search_random_forest(
    X_train, y_train, X_test, y_test, numeric_cols, categorical_cols, random_state: int
) -> SearchResult:
    model_params = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        # sklearn>=1.4 deprecates 'auto'; use 'sqrt' for stable behavior.
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": random_state,
    }
    # Train/validate
    train_pre = build_preprocessor(numeric_cols, categorical_cols)
    train_model = RandomForestRegressor(**model_params)
    train_pipe = Pipeline([("preprocessor", train_pre), ("model", train_model)])
    train_pipe.fit(X_train, y_train)
    metrics = _score_model(train_pipe, X_test, y_test)

    # Refit on full data for artifact
    full_pre = build_preprocessor(numeric_cols, categorical_cols)
    full_model = RandomForestRegressor(**model_params)
    full_pipe = Pipeline([("preprocessor", full_pre), ("model", full_model)])
    full_pipe.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))

    params = {"n_estimators": model_params["n_estimators"], "min_samples_leaf": model_params["min_samples_leaf"]}
    return SearchResult(name="RandomForest", pipeline=full_pipe, holdout_metrics=metrics, params=params)


def _search_lightgbm(
    X_train, y_train, X_test, y_test, numeric_cols, categorical_cols, random_state: int, n_trials: int
) -> Optional[SearchResult]:
    try:
        import lightgbm as lgb  # type: ignore
    except ImportError:
        return None

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.2),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.8),
        }
        pre = build_preprocessor(numeric_cols, categorical_cols)
        pre.fit(X_train, y_train)
        constraints = _compute_monotone_constraints(pre, "Age")
        model = lgb.LGBMRegressor(
            random_state=random_state,
            n_jobs=-1,
            **params,
        )
        if constraints:
            model.set_params(monotone_constraints=constraints)
        pipe = Pipeline([("preprocessor", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        return mean_absolute_error(y_test, preds)

    if optuna is None:
        # Fallback to a single fit with conservative params.
        base_params = dict(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=127,
            max_depth=8,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
        train_pre = build_preprocessor(numeric_cols, categorical_cols)
        train_pre.fit(X_train, y_train)
        constraints = _compute_monotone_constraints(train_pre, "Age")
        model = lgb.LGBMRegressor(**base_params)
        if constraints:
            model.set_params(monotone_constraints=constraints)
        train_pipe = Pipeline([("preprocessor", train_pre), ("model", model)])
        train_pipe.fit(X_train, y_train)
        metrics = _score_model(train_pipe, X_test, y_test)

        # Refit on full data
        full_pre = build_preprocessor(numeric_cols, categorical_cols)
        full_pre.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
        full_constraints = _compute_monotone_constraints(full_pre, "Age")
        full_model = lgb.LGBMRegressor(**base_params)
        if full_constraints:
            full_model.set_params(monotone_constraints=full_constraints)
        full_pipe = Pipeline([("preprocessor", full_pre), ("model", full_model)])
        full_pipe.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
        return SearchResult(
            name="LightGBM",
            pipeline=full_pipe,
            holdout_metrics=metrics,
            params=base_params,
        )

    study = optuna.create_study(direction="minimize", study_name="lgbm_mae")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params

    full_X = pd.concat([X_train, X_test])
    full_y = pd.concat([y_train, y_test])

    # Train/validate metrics on train-only fit
    train_pre = build_preprocessor(numeric_cols, categorical_cols)
    train_pre.fit(X_train, y_train)
    train_constraints = _compute_monotone_constraints(train_pre, "Age")
    train_model = lgb.LGBMRegressor(random_state=random_state, n_jobs=-1, **best_params)
    if train_constraints:
        train_model.set_params(monotone_constraints=train_constraints)
    train_pipe = Pipeline([("preprocessor", train_pre), ("model", train_model)])
    train_pipe.fit(X_train, y_train)
    metrics = _score_model(train_pipe, X_test, y_test)

    # Refit on full data for artifact
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(full_X, full_y)
    constraints = _compute_monotone_constraints(preprocessor, "Age")
    best_model = lgb.LGBMRegressor(random_state=random_state, n_jobs=-1, **best_params)
    if constraints:
        best_model.set_params(monotone_constraints=constraints)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", best_model)])
    pipeline.fit(full_X, full_y)
    return SearchResult(name="LightGBM", pipeline=pipeline, holdout_metrics=metrics, params=best_params)


def _search_xgboost(
    X_train, y_train, X_test, y_test, numeric_cols, categorical_cols, random_state: int, n_trials: int
) -> Optional[SearchResult]:
    try:
        from xgboost import XGBRegressor  # type: ignore
    except ImportError:
        return None

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 6.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "gamma": trial.suggest_float("gamma", 0.0, 0.3),
        }
        pre = build_preprocessor(numeric_cols, categorical_cols)
        pipe = Pipeline(
            [
                ("preprocessor", pre),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=random_state,
                        n_jobs=-1,
                        tree_method="hist",
                        **params,
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        return mean_absolute_error(y_test, preds)

    if optuna is None:
        model_params = dict(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.9,
            colsample_bytree=0.8,
        )
        train_pre = build_preprocessor(numeric_cols, categorical_cols)
        train_model = XGBRegressor(**model_params)
        train_pipe = Pipeline([("preprocessor", train_pre), ("model", train_model)])
        train_pipe.fit(X_train, y_train)
        metrics = _score_model(train_pipe, X_test, y_test)

        full_pre = build_preprocessor(numeric_cols, categorical_cols)
        full_model = XGBRegressor(**model_params)
        full_pipe = Pipeline([("preprocessor", full_pre), ("model", full_model)])
        full_pipe.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
        return SearchResult(name="XGBoost", pipeline=full_pipe, holdout_metrics=metrics, params=model_params)

    study = optuna.create_study(direction="minimize", study_name="xgb_mae")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params

    full_X = pd.concat([X_train, X_test])
    full_y = pd.concat([y_train, y_test])

    # Train/validate metrics on train-only fit
    train_pre = build_preprocessor(numeric_cols, categorical_cols)
    train_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        **best_params,
    )
    train_pipe = Pipeline([("preprocessor", train_pre), ("model", train_model)])
    train_pipe.fit(X_train, y_train)
    metrics = _score_model(train_pipe, X_test, y_test)

    # Refit on full data
    pre = build_preprocessor(numeric_cols, categorical_cols)
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        **best_params,
    )
    pipe = Pipeline([("preprocessor", pre), ("model", model)])
    pipe.fit(full_X, full_y)
    return SearchResult(name="XGBoost", pipeline=pipe, holdout_metrics=metrics, params=best_params)


def _search_catboost(
    X_train, y_train, X_test, y_test, numeric_cols, categorical_cols, random_state: int, n_trials: int
) -> Optional[SearchResult]:
    try:
        from catboost import CatBoostRegressor  # type: ignore
    except ImportError:
        return None

    # CatBoost handles categoricals internally; here we still OHE for simplicity/compatibility.
    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 6, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "iterations": trial.suggest_int("iterations", 400, 900),
        }
        pre = build_preprocessor(numeric_cols, categorical_cols)
        model = CatBoostRegressor(
            random_seed=random_state,
            loss_function="RMSE",
            verbose=False,
            **params,
        )
        pipe = Pipeline([("preprocessor", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        return mean_absolute_error(y_test, preds)

    if optuna is None:
        base_params = dict(
            random_seed=random_state,
            loss_function="RMSE",
            verbose=False,
            depth=8,
            learning_rate=0.05,
            iterations=600,
            l2_leaf_reg=3.0,
        )
        train_pre = build_preprocessor(numeric_cols, categorical_cols)
        train_model = CatBoostRegressor(**base_params)
        train_pipe = Pipeline([("preprocessor", train_pre), ("model", train_model)])
        train_pipe.fit(X_train, y_train)
        metrics = _score_model(train_pipe, X_test, y_test)

        full_pre = build_preprocessor(numeric_cols, categorical_cols)
        full_model = CatBoostRegressor(**base_params)
        full_pipe = Pipeline([("preprocessor", full_pre), ("model", full_model)])
        full_pipe.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
        return SearchResult(name="CatBoost", pipeline=full_pipe, holdout_metrics=metrics, params=base_params)

    study = optuna.create_study(direction="minimize", study_name="catboost_mae")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params

    full_X = pd.concat([X_train, X_test])
    full_y = pd.concat([y_train, y_test])

    # Train/validate metrics on train-only fit
    train_pre = build_preprocessor(numeric_cols, categorical_cols)
    train_model = CatBoostRegressor(random_seed=random_state, loss_function="RMSE", verbose=False, **best_params)
    train_pipe = Pipeline([("preprocessor", train_pre), ("model", train_model)])
    train_pipe.fit(X_train, y_train)
    metrics = _score_model(train_pipe, X_test, y_test)

    # Refit on full data
    pre = build_preprocessor(numeric_cols, categorical_cols)
    model = CatBoostRegressor(random_seed=random_state, loss_function="RMSE", verbose=False, **best_params)
    pipe = Pipeline([("preprocessor", pre), ("model", model)])
    pipe.fit(full_X, full_y)
    return SearchResult(name="CatBoost", pipeline=pipe, holdout_metrics=metrics, params=best_params)


def run_model_search(
    df_raw, config: TrainingConfig, n_trials: int = 20, test_size: float | None = None
) -> Tuple[List[SearchResult], SearchResult]:
    """Run candidate searches and return leaderboard + best result."""
    df = clean_dataframe(df_raw, config)
    X, y = split_features_target(df, config.target)
    numeric_cols, categorical_cols = infer_feature_types(df, config)

    X, y = shuffle(X, y, random_state=config.random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size or config.test_size,
        random_state=config.random_state,
        stratify=df["Position"],
    )

    results: List[SearchResult] = []

    # RandomForest baseline
    results.append(_search_random_forest(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols, config.random_state))

    # LightGBM
    lgbm_res = _search_lightgbm(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols, config.random_state, n_trials)
    if lgbm_res:
        results.append(lgbm_res)

    # XGBoost
    xgb_res = _search_xgboost(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols, config.random_state, n_trials)
    if xgb_res:
        results.append(xgb_res)

    # CatBoost
    cat_res = _search_catboost(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols, config.random_state, n_trials)
    if cat_res:
        results.append(cat_res)

    if not results:
        raise RuntimeError("No models were trained; ensure boosting libraries are installed.")

    # Pick best by MAE.
    best = sorted(results, key=lambda r: r.holdout_metrics["mae"])[0]
    return results, best
