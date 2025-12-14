from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .torch_model import TabularNet
from .features import clean_dataframe, infer_feature_types, split_features_target
from .config import TrainingConfig


class TorchPredictor:
    def __init__(self, model: TabularNet, meta: Dict, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.meta = meta
        self.device = device

    @classmethod
    def load(cls, model_dir: Path):
        meta = json.loads((model_dir / "meta.json").read_text())
        model = TabularNet(
            cat_cardinalities=[len(meta["cat_maps"][c]) for c in meta["cat_cols"]],
            embed_dim=meta["embed_dim"],
            num_features=len(meta["num_cols"]),
            hidden=meta["hidden"],
        )
        state = torch.load(model_dir / "model.pth", map_location="cpu")
        model.load_state_dict(state)
        return cls(model=model, meta=meta)

    def _encode(self, df):
        cat_cols = self.meta["cat_cols"]
        num_cols = self.meta["num_cols"]
        cat_maps = self.meta["cat_maps"]
        num_mean = self.meta["num_mean"]
        num_std = self.meta["num_std"]
        cats = []
        for col in cat_cols:
            mapping = cat_maps[col]
            cats.append(
                df[col]
                .astype(str)
                .replace("nan", "<UNK>")
                .apply(lambda v: mapping.get(v, mapping.get("<UNK>", 0)))
                .to_numpy(dtype=np.int64)
            )
        cats = np.stack(cats, axis=1) if cats else np.zeros((len(df), 0), dtype=np.int64)
        nums = df[num_cols].copy()
        for c in num_cols:
            nums[c] = (nums[c] - num_mean[c]) / (num_std[c] if num_std[c] != 0 else 1.0)
        nums = nums.fillna(0).to_numpy(dtype=np.float32)
        return torch.tensor(cats, device=self.device), torch.tensor(nums, device=self.device)

    def predict_df(self, df) -> np.ndarray:
        with torch.no_grad():
            cats, nums = self._encode(df)
            preds = self.model(cats, nums).cpu().numpy()
        return preds


def load_torch_predictor(model_root: Path) -> TorchPredictor | None:
    try:
        return TorchPredictor.load(model_root)
    except Exception:
        return None
