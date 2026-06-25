"""
Unified metrics computation entry point.

Aggregates PSNR, SSIM, SAM, RMSE, NDVI-MAE over a dataset and saves
a JSON report.

Usage:
    python -m ai.metrics.compute \
        --pred outputs/predictions \
        --target datasets/sen2cr/clear \
        --mask  datasets/sen2cr/masks \
        --output outputs/metrics/report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai.metrics.ndvi_mae import ndvi_mae
from ai.metrics.psnr import psnr
from ai.metrics.sam import sam
from ai.metrics.ssim import ssim


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root-mean-square error."""
    return torch.sqrt(torch.nn.functional.mse_loss(pred, target))


def compute_all(
    pred: torch.Tensor,
    target: torch.Tensor,
    cloudy: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    """
    Compute all metrics for a single batch/image.

    Args:
        pred:   [B, 3, H, W] predictions.
        target: [B, 3, H, W] ground truth.
        cloudy: Optional [B, 3, H, W] cloudy input (for NDVI baseline).
        mask:   Optional [B, 1, H, W] cloud mask.

    Returns:
        Dictionary of metric → value.
    """
    with torch.no_grad():
        out: Dict[str, float] = {
            "psnr_db": psnr(pred, target).item(),
            "ssim": ssim(pred, target).item(),
            "sam_rad": sam(pred, target).item(),
            "sam_deg": sam(pred, target, degrees=True).item(),
            "rmse": rmse(pred, target).item(),
            "ndvi_mae_global": ndvi_mae(pred, target).item(),
        }
        if mask is not None:
            out["ndvi_mae_cloud"] = ndvi_mae(pred, target, mask=mask).item()
        if cloudy is not None and mask is not None:
            baseline = ndvi_mae(cloudy, target, mask=mask).item()
            out["ndvi_improvement"] = baseline - out["ndvi_mae_cloud"]
    return out


def evaluate_directory(
    pred_dir: Path,
    target_dir: Path,
    mask_dir: Path | None,
    output_path: Path,
) -> Dict[str, float]:
    """
    Evaluate all .npy predictions in ``pred_dir`` against targets.

    Args:
        pred_dir:    Directory of prediction .npy files.
        target_dir:  Directory of ground-truth .npy files.
        mask_dir:    Optional directory of mask .npy files.
        output_path: Where to write the JSON report.

    Returns:
        Aggregated metrics dict.
    """
    pred_files = sorted(pred_dir.glob("*.npy"))
    if not pred_files:
        raise RuntimeError(f"No .npy files found in {pred_dir}")

    all_metrics: list[Dict[str, float]] = []

    for pred_path in pred_files:
        stem = pred_path.stem
        tgt_path = target_dir / f"{stem}.npy"
        if not tgt_path.exists():
            continue

        pred_np = np.load(str(pred_path)).astype(np.float32)
        tgt_np = np.load(str(tgt_path)).astype(np.float32)

        # [H, W, 3] → [1, 3, H, W]
        p_t = torch.from_numpy(pred_np.transpose(2, 0, 1)).unsqueeze(0)
        t_t = torch.from_numpy(tgt_np.transpose(2, 0, 1)).unsqueeze(0)

        mask_t: torch.Tensor | None = None
        if mask_dir is not None:
            msk_path = mask_dir / f"{stem}.npy"
            if msk_path.exists():
                msk_np = np.load(str(msk_path)).astype(np.float32)
                mask_t = torch.from_numpy(msk_np).unsqueeze(0).unsqueeze(0)

        metrics = compute_all(p_t, t_t, mask=mask_t)
        all_metrics.append(metrics)

    if not all_metrics:
        raise RuntimeError("No matching prediction/target pairs found.")

    # Average over all samples
    keys = all_metrics[0].keys()
    aggregated = {k: float(sum(m[k] for m in all_metrics) / len(all_metrics)) for k in keys}
    aggregated["n_samples"] = float(len(all_metrics))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    return aggregated


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate cloud removal metrics")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--output", default="outputs/metrics/report.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metrics = evaluate_directory(
        pred_dir=Path(args.pred),
        target_dir=Path(args.target),
        mask_dir=Path(args.mask) if args.mask else None,
        output_path=Path(args.output),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
