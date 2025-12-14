import sys
from pathlib import Path
import json

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

# Add root to path so we can import app
sys.path.append(str(PROJECT_ROOT))

from app import predict_with_adjustments, get_data

def main():
    print("Verifying uncertainty...")
    df = get_data()
    if df.empty:
        print("No data found")
        return
    
    # Pick a random player
    player_id = int(df.iloc[0]["ID"])
    print(f"Testing with Player ID: {player_id}")
    
    # Test Normal Prediction
    try:
        res = predict_with_adjustments(player_id, {})
        print("Result keys:", res.keys())
        if "baseline_uncertainty" in res:
            print("Baseline Uncertainty:", json.dumps(res["baseline_uncertainty"], indent=2))
        else:
            print("FAILED: baseline_uncertainty missing")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # Test Extrapolation (Change Age to 100)
    print("\nTesting Extrapolation (Age=100)...")
    try:
        res = predict_with_adjustments(player_id, {"Age": 100})
        unc = res.get("adjusted_uncertainty")
        if unc:
            print("Adjusted Uncertainty:", json.dumps(unc, indent=2))
            if unc.get("is_extrapolating"):
                print("SUCCESS: Extrapolation detected!")
            else:
                print("WARNING: Extrapolation NOT detected for Age=100")
        else:
            print("FAILED: adjusted_uncertainty missing")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
