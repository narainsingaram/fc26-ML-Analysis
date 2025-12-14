import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT))

from app import get_model, get_data, get_config, clean_dataframe, get_torch_predictor

def main():
    print("Debugging What-If Logic...")
    
    # 1. Load Data
    df = get_data()
    # Pick a player that might yield ~72 OVR range or just a random one. 
    # Let's try to find a player with low OVR to see if we can boost them.
    player = df[df["OVR"] < 70].iloc[0]
    print(f"Base Player: {player['Name']} (ID: {player['ID']}) - Base OVR: {player['OVR']}")
    
    # 2. Define Adjustments from User
    # Reactions 89, SHO 90, PHY 85, PAC 88, Shot Power 91, Sprint Speed 92, Ball Control 88, Strength 82
    adjustments = {
        "Reactions": 89,
        "SHO": 90,
        "PHY": 85,
        "PAC": 88,
        "Shot Power": 91,
        "Sprint Speed": 92,
        "Ball Control": 88,
        "Strength": 82,
        # Hypothesis: CM needs Passing
        "Short Passing": 90,
        "Long Passing": 90,
        "Vision": 90,
        "PAS": 90
    }
    
    # 3. Predict Baseline
    cfg = get_config()
    row = df[df["ID"] == player["ID"]]
    cleaned = clean_dataframe(row, cfg, drop_null_only=False)
    X_base = cleaned.drop(columns=[cfg.target], errors="ignore")
    
    model = get_model()
    torch_model = get_torch_predictor()
    
    if torch_model:
        base_pred = float(torch_model.predict_df(X_base)[0])
        print(f"Model: Torch (Base Pred: {base_pred:.2f})")
    else:
        base_pred = float(model.predict(X_base)[0])
        print(f"Model: Sklearn (Base Pred: {base_pred:.2f})")

    # 4. Predict Adjusted
    adjusted = cleaned.copy()
    for col, val in adjustments.items():
        if col in adjusted.columns:
            adjusted[col] = val
        else:
            print(f"Warning: Column '{col}' not found in dataframe columns")
            
    # Test 3: Set ALL relevant numeric stats to 90 + Fix League/Age
    print("\nTest 3: Setting ALL numeric stats to 90 + League='Premier League' + Age=27...")
    all_90 = cleaned.copy()
    numeric_cols = all_90.select_dtypes(include=[np.number]).columns
    count_changed = 0
    for c in numeric_cols:
        if c not in ["ID", "height_cm", "weight_kg", "Weak foot", "Skill moves", "GENDER"]: 
             all_90[c] = 90
             count_changed += 1
    
    # Set Age explicitly
    all_90["Age"] = 27
    all_90["age_bucket"] = 27 # Approximation
    
    # Set League if column exists
    if "League" in all_90.columns:
        # Find a top league from the data to be sure
        top_leagues = df["League"].value_counts().head(5).index.tolist()
        target_league = "Premier League" if "Premier League" in top_leagues else top_leagues[0]
        print(f"Setting League to: {target_league}")
        all_90["League"] = target_league
    
    X_90 = all_90.drop(columns=[cfg.target], errors="ignore")
    if torch_model:
        pred_90 = float(torch_model.predict_df(X_90)[0])
    else:
        pred_90 = float(model.predict(X_90)[0])
    print(f"Optimized All-90 Pred: {pred_90:.2f}")

    # Analysis of the User's Case
    print("\nAnalysis:")
    user_keys = adjustments.keys()
    print(f"User changed {len(user_keys)} stats.")
    
    # Calculate average of UNCHANGED stats
    unchanged_vals = []
    for c in numeric_cols:
        if c not in ["ID", "Age", "age_bucket"] and c not in user_keys:
             val = cleaned.iloc[0].get(c)
             if isinstance(val, (int, float)):
                 unchanged_vals.append(val)
    
    if unchanged_vals:
        print(f"Average of {len(unchanged_vals)} UNCHANGED stats: {sum(unchanged_vals)/len(unchanged_vals):.1f}")
    print(f"Player Position: {player.get('Position')}")
    print("Features used in model:", list(X_base.columns))
    
    # Check if Position is affecting it. 
    # Usually Rating Engines rely heavily on position-specific stats.
    # e.g. for a CB, 'SHO' might have 0 weight.
    
    # Let's try to change ALL stats to 99 to see max potential
    all_99 = cleaned.copy()
    numeric_cols = all_99.select_dtypes(include=[np.number]).columns
    for c in numeric_cols:
        if c != "ID":
            all_99[c] = 99
    X_99 = all_99.drop(columns=[cfg.target], errors="ignore")
    if torch_model:
        max_pred = float(torch_model.predict_df(X_99)[0])
    else:
        max_pred = float(model.predict(X_99)[0])
    print(f"Theoretical Max (All stats 99): {max_pred:.2f}")

if __name__ == "__main__":
    main()
