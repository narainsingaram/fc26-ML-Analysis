import sys
from pathlib import Path
import pandas as pd

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from rating_engine.data import load_player_data
from rating_engine.config import DEFAULT_CONFIG

def main():
    df = load_player_data(DEFAULT_CONFIG.data_path, DEFAULT_CONFIG.age_bins)
    print("Columns:", list(df.columns))
    print("\nSample Data:")
    print(df[["Name", "Height", "Weight", "Weak foot", "Skill moves", "PlayStyles", "Attacking work rate", "Defensive work rate"]].head(3))

if __name__ == "__main__":
    main()
