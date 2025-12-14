#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rating_engine.config import DEFAULT_CONFIG, TrainingConfig
from rating_engine.data import load_player_data
from rating_engine.features import build_preprocessor, clean_dataframe, infer_feature_types


def quantile_loss():
    import tensorflow as tf
    qs = [0.1, 0.5, 0.9]

    def loss(y_true, y_pred):
        e = y_true - y_pred
        losses = []
        for i, q in enumerate(qs):
            ei = e[:, i]
            losses.append(tf.reduce_mean(tf.maximum(q * ei, (q - 1) * ei)))
        return tf.add_n(losses) / len(qs)

    return loss


def main():
    parser = argparse.ArgumentParser(description="Train quantile regression model for OVR (p10, p50, p90).")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_CONFIG.data_path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/quantile_model"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        raise SystemExit("TensorFlow is required for quantile training. Please install tensorflow.")

    config = TrainingConfig(data_path=args.data_path)
    df_raw = load_player_data(config.data_path, config.age_bins)
    df = clean_dataframe(df_raw, config)  # drop all-null cols
    X = df.drop(columns=[config.target], errors="ignore")
    y = df[config.target].to_numpy()

    # Preprocess
    num_cols, cat_cols = infer_feature_types(df, config)[0], infer_feature_types(df, config)[1]
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_mat = preprocessor.fit_transform(X)
    if hasattr(X_mat, "toarray"):
        X_mat = X_mat.toarray()

    input_dim = X_mat.shape[1]

    import tensorflow as tf

    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(3)(x)  # p10, p50, p90
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=quantile_loss(),
    )

    # y as (n,3) replicated for quantiles
    y_mat = np.stack([y, y, y], axis=1)
    model.fit(X_mat, y_mat, epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "model.keras")
    joblib.dump(preprocessor, args.output_dir / "preprocessor.joblib")
    print(f"Saved quantile model and preprocessor to {args.output_dir}")


if __name__ == "__main__":
    main()
