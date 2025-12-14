from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set


@dataclass
class TrainingConfig:
    data_path: Path = Path("data/raw/EAFC26.csv")
    target: str = "OVR"
    drop_columns: Set[str] = field(
        default_factory=lambda: {
            "Name",
            "Rank",
            "ID",
            "url",
            "card",
            "Alternative positions",
            "play style",
        }
    )
    categorical_columns: List[str] = field(
        default_factory=lambda: [
            "GENDER",
            "Position",
            "Preferred foot",
            "League",
            "Team",
            "Nation",
        ]
    )
    test_size: float = 0.2
    random_state: int = 42
    output_dir: Path = Path("artifacts")
    top_leagues: int = 15
    age_bins: List[int] = field(default_factory=lambda: [15, 20, 24, 27, 30, 33, 36, 50])


DEFAULT_CONFIG = TrainingConfig()
