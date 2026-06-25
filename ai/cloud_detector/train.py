"""
Training script for the lightweight U-Net cloud detector.

Usage:
    python -m ai.cloud_detector.train \
        --data datasets/sen2cr \
        --epochs 50 \
        --batch-size 8 \
        --lr 3e-4 \
        --output ai/cloud_detector/checkpoints
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai.cloud_detector.dataset import CloudDetectionDataset
from ai.cloud_detector.model import CloudDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def train(
    data_root: str,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 3e-4,
    output_dir: str = "ai/cloud_detector/checkpoints",
    device_str: str = "auto",
) -> None:
    device = torch.device(
        "cuda" if (device_str == "auto" and torch.cuda.is_available()) else device_str
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    full_ds = CloudDetectionDataset(data_root, patch_size=256, augment=True)
    val_n = max(1, int(len(full_ds) * 0.1))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_n, val_n])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    model = CloudDetector(in_channels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_losses: list[float] = []
        for batch in train_dl:
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)
            pred = model(img)
            loss = criterion(pred, msk)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_dl:
                img = batch["image"].to(device)
                msk = batch["mask"].to(device)
                pred = model(img)
                val_losses.append(criterion(pred, msk).item())

        mean_train = sum(train_losses) / len(train_losses)
        mean_val = sum(val_losses) / len(val_losses)
        log.info("Epoch %d | Train %.4f | Val %.4f", epoch + 1, mean_train, mean_val)

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save(model.state_dict(), out / "best.pt")

    torch.save(model.state_dict(), out / "last.pt")
    log.info("Training complete. Best val loss: %.4f", best_val_loss)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output", default="ai/cloud_detector/checkpoints")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(args.data, args.epochs, args.batch_size, args.lr, args.output)
