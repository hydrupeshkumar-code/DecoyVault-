"""
Evaluation script for the cloud detector.

Reports IoU, F1-score, precision, recall, and accuracy on a test split.

Usage:
    python -m ai.cloud_detector.evaluate \
        --data datasets/sen2cr \
        --checkpoint ai/cloud_detector/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai.cloud_detector.dataset import CloudDetectionDataset
from ai.cloud_detector.model import CloudDetector


def _iou(pred_bin: torch.Tensor, target: torch.Tensor) -> float:
    intersection = (pred_bin * target).sum()
    union = (pred_bin + target).clamp(max=1).sum()
    return (intersection / union.clamp(min=1e-8)).item()


def evaluate(data_root: str, checkpoint: str, threshold: float = 0.5) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CloudDetector(in_channels=3).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    ds = CloudDetectionDataset(data_root, augment=False)
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2)

    tp = tn = fp = fn = 0
    ious: list[float] = []

    with torch.no_grad():
        for batch in dl:
            img = batch["image"].to(device)
            msk = batch["mask"].to(device)
            pred = (model(img) > threshold).float()

            tp += (pred * msk).sum().item()
            tn += ((1 - pred) * (1 - msk)).sum().item()
            fp += (pred * (1 - msk)).sum().item()
            fn += ((1 - pred) * msk).sum().item()
            ious.append(_iou(pred, msk))

    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1e-8)
    mean_iou = sum(ious) / len(ious)

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "mean_iou": mean_iou,
    }
    for k, v in results.items():
        print(f"{k:15s}: {v:.4f}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()
    evaluate(args.data, args.checkpoint, args.threshold)
