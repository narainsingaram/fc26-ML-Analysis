import sys
from pathlib import Path
import json
import pandas as pd

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT))
from app import app

def main():
    print("Verifying Advanced Search Endpoint...")
    
    with app.test_client() as client:
        # Test 1: Physical (Tall)
        resp = client.get('/api/players?min_height=195')
        data = resp.get_json()
        players = data.get('players', [])
        print(f"\nTest 1 (Height >= 195cm): Found {len(players)}")
        if players:
            print(f"Sample: {players[0]['Name']} ({players[0]['height_cm']}cm)")
            if all(p['height_cm'] >= 195 for p in players):
                 print("SUCCESS: All players >= 195cm")
            else:
                 print("FAILURE: Height filter mismatch")
        
        # Test 2: Complex (Brazil + 5 Star Skills)
        resp = client.get('/api/players?nation=Brazil&min_sm=5')
        data = resp.get_json()
        players = data.get('players', [])
        print(f"\nTest 2 (Brazil + 5★ Skills): Found {len(players)}")
        if players:
            print(f"Sample: {players[0]['Name']} (Nation: {players[0]['Nation']}, SM: {players[0]['Skill moves']})")
            if all(p['Nation'] == "Brazil" and p['Skill moves'] >= 5 for p in players):
                 print("SUCCESS: All players correspond to filters.")
            else:
                 print("FAILURE: Nation/Skills filter mismatch")

        # Test 3: Playstyle (Finesse)
        # Note: 'play style' is the column, 'playstyle' is the param
        resp = client.get('/api/players?playstyle=Finesse')
        data = resp.get_json()
        players = data.get('players', [])
        print(f"\nTest 3 (PlayStyle 'Finesse'): Found {len(players)}")
        if players:
            # Check a few
            sample = players[0]
            print(f"Sample: {sample['Name']} Playstyles: {sample.get('play style', '')[:50]}...")
            if 'finesse' in str(sample.get('play style', '')).lower():
                 print("SUCCESS: Playstyle found.")
            else:
                 print("FAILURE: Playstyle mismatch.")

if __name__ == "__main__":
    main()
