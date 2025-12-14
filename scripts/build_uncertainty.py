#!/usr/bin/env python3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Ensure src is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from rating_engine.config import DEFAULT_CONFIG, TrainingConfig
from rating_engine.data import load_player_data
from rating_engine.features import build_preprocessor, clean_dataframe, infer_feature_types
from rating_engine.quantile_tf import load_quantile_predictor

def main():
    print("Building Uncertainty Model...")
    
    # 1. Setup & Load Data
    output_dir = Path("artifacts/quantile_model")
    if not output_dir.exists():
        print(f"Error: {output_dir} does not exist. Train quantile model first.")
        sys.exit(1)

    config = DEFAULT_CONFIG
    df_raw = load_player_data(config.data_path, config.age_bins)
    df = clean_dataframe(df_raw, config)
    
    # We use the same features as the model
    X = df.drop(columns=[config.target], errors="ignore")
    
    # 2. Build Preprocessor for Distance Metric
    print("Fitting preprocessor...")
    num_cols, cat_cols = infer_feature_types(df, config)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_trans = preprocessor.fit_transform(X)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    # 3. Train KNN for Density/Distance
    print("Fitting KNN for distance estimation...")
    # k=5 is a standard choice for density estimation
    knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    knn.fit(X_trans)
    
    # 4. Calculate distributions on Training Set
    print("Calculating training set statistics using 5-fold CV logic or just plain training set (using plain for reference)...")
    # For thresholds, we want to know what "normal" looks like.
    dists, _ = knn.kneighbors(X_trans)
    mean_dists = dists.mean(axis=1) # Average distance to 5 nearest neighbors
    
    dist_threshold = np.percentile(mean_dists, 95)
    print(f"Distance Threshold (95th): {dist_threshold:.4f}")
    
    # 5. Load Quantile Model and calculate width thresholds
    print("Loading quantile model to calibrate uncertainty widths...")
    q_model = load_quantile_predictor(output_dir)
    if not q_model:
        print("Error: Could not load quantile model.")
        sys.exit(1)
        
    # We predict on the whole training set to see the distribution of widths
    # ( Ideally we would use a holdout set, but for defining 'high uncertainty' relative to typical data, this is acceptable)
    preds = q_model.predict(X) # (N, 3)
    p10 = preds[:, 0]
    p90 = preds[:, 2]
    widths = p90 - p10
    
    width_threshold = np.percentile(widths, 95)
    print(f"Width Threshold (95th): {width_threshold:.4f}")

    # 6. Save Artifacts
    print("Saving uncertainty artifacts...")
    joblib.dump(knn, output_dir / "uncertainty_knn.joblib")
    joblib.dump(preprocessor, output_dir / "uncertainty_preprocessor.joblib")
    joblib.dump({
        "dist_threshold": dist_threshold,
        "width_threshold": width_threshold
    }, output_dir / "uncertainty_meta.joblib")
    
    print("Done! Uncertainty model ready.")

if __name__ == "__main__":
    main()
