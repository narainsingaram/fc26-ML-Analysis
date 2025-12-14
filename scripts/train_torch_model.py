#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from rating_engine.config import DEFAULT_CONFIG, TrainingConfig  # noqa: E402
from rating_engine.data import load_player_data  # noqa: E402
from rating_engine.features import clean_dataframe, infer_feature_types, split_features_target  # noqa: E402
from rating_engine.torch_model import TabularNet  # noqa: E402


class PlayerDataset(Dataset):
    def __init__(self, cats, nums, targets):
        self.cats = cats
        self.nums = nums
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.cats[idx], self.nums[idx], self.targets[idx]


def build_encoders(df, cat_cols, num_cols):
    cat_maps: Dict[str, Dict[str, int]] = {}
    for col in cat_cols:
        series = df[col].astype(str).replace("nan", "<UNK>")
        uniq = series.unique().tolist()
        mapping = {v: i for i, v in enumerate(["<UNK>"] + sorted(uniq))}
        cat_maps[col] = mapping
    num_mean = {c: float(df[c].mean()) for c in num_cols}
    num_std = {c: float(df[c].std() if df[c].std() > 1e-6 else 1.0) for c in num_cols}
    return cat_maps, num_mean, num_std


def encode_df(df, cat_cols, num_cols, cat_maps, num_mean, num_std):
    cats = []
    for col in cat_cols:
        mapping = cat_maps[col]
        series = df[col].astype(str).replace("nan", "<UNK>")
        vals = series.apply(lambda v: mapping.get(v, mapping["<UNK>"]))
        cats.append(vals.to_numpy(dtype=np.int64))
    cats = np.stack(cats, axis=1) if cats else np.zeros((len(df), 0), dtype=np.int64)
    nums = df[num_cols].fillna(0)
    for c in num_cols:
        nums[c] = (nums[c] - num_mean[c]) / num_std[c]
    return cats, nums.to_numpy(dtype=np.float32)


def train_model(config: TrainingConfig, epochs: int = 10, batch_size: int = 256, lr: float = 1e-3):
    df_raw = load_player_data(config.data_path, config.age_bins)
    df = clean_dataframe(df_raw, config)
    cat_cols, num_cols = infer_feature_types(df, config)[1], infer_feature_types(df, config)[0]
    X, y = split_features_target(df, config.target)

    cat_maps, num_mean, num_std = build_encoders(df, cat_cols, num_cols)
    cats_np, nums_np = encode_df(df, cat_cols, num_cols, cat_maps, num_mean, num_std)

    dataset = PlayerDataset(torch.tensor(cats_np), torch.tensor(nums_np), torch.tensor(y.to_numpy(), dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 8
    hidden = [128, 64]
    model = TabularNet(cat_cardinalities=[len(cat_maps[c]) for c in cat_cols], embed_dim=embed_dim, num_features=len(num_cols), hidden=hidden)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for cats, nums, targets in loader:
            cats, nums, targets = cats.to(device), nums.to(device), targets.to(device)
            opt.zero_grad()
            preds = model(cats, nums)
            loss = loss_fn(preds, targets)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(targets)
        print(f"Epoch {epoch+1}/{epochs} loss={total_loss/len(dataset):.4f}")

    # Save artifacts
    out_dir = config.output_dir / "torch_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pth")
    meta = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_maps": cat_maps,
        "num_mean": num_mean,
        "num_std": num_std,
        "embed_dim": embed_dim,
        "hidden": hidden,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta))
    print(f"Saved torch model to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train a PyTorch position-aware tabular model for OVR.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_CONFIG.data_path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    config = TrainingConfig(data_path=args.data_path, output_dir=args.output_dir)
    train_model(config, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)


if __name__ == "__main__":
    main()
