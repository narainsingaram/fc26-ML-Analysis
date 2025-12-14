import sys
from pathlib import Path
import json

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT))

from app import predict_with_adjustments, get_data

def main():
    print("Verifying Residual Calibration...")
    df = get_data()
    
    # Pick a player with a known gap (e.g. Pred=53, Actual=69)
    # Finding Marko Bulat 261143 from previous debug output
    player_id = 261143
    row = df[df["ID"] == player_id]
    if row.empty:
        # Fallback to first player
        player_id = int(df.iloc[0]["ID"])
        row = df.iloc[[0]]
        
    actual = float(row.iloc[0]["OVR"])
    print(f"Player: {row.iloc[0]['Name']} (ID: {player_id})")
    print(f"Actual OVR: {actual}")

    # 1. Baseline Call (No adjustments)
    res = predict_with_adjustments(player_id, {})
    
    baseline_pred = res["baseline"]
    print(f"Calibrated Baseline: {baseline_pred:.2f}")
    
    # Check if baseline matches actual (tolerance 0.01)
    if abs(baseline_pred - actual) < 0.1:
        print("SUCCESS: Baseline is calibrated to Actual OVR.")
    else:
        print(f"FAILURE: Baseline {baseline_pred:.2f} != Actual {actual}")

    # 2. Adjustment Call (Boost stats)
    # Small boost
    adj_stats = {"Reactions": 99}
    res_adj = predict_with_adjustments(player_id, adj_stats)
    
    adj_pred = res_adj["adjusted"]
    delta = res_adj["delta"]
    
    print(f"Adjusted (Reactions=99): {adj_pred:.2f}")
    print(f"Delta: {delta:.2f}")
    
    # Verify Delta Consistency
    # The delta should be (Model_Adj - Model_Base), which is preserved even after calibration
    # Adjusted_Calibrated = Baseline_Calibrated + Delta
    calc_adj = baseline_pred + delta
    if abs(calc_adj - adj_pred) < 0.01:
         print("SUCCESS: Delta consistency maintained.")
    else:
         print(f"FAILURE: Delta incosistency. {calc_adj} != {adj_pred}")

if __name__ == "__main__":
    main()
