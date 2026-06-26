"""
Single-image inference with a trained Restormer model.

Usage:
    python -m ai.restormer.infer \
        --checkpoint ai/restormer/checkpoints/phase2_final.pt \
        --input path/to/cloudy.npy \
        --mask  path/to/mask.npy \
        --output path/to/output.npy \
        [--config ai/restormer/train_config.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ai.restormer.model import Restormer


def load_model(
    checkpoint_path: str | Path,
    config_path: str | Path | None,
    device: torch.device,
) -> Restormer:
    """Load a Restormer model from a checkpoint."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        mcfg = cfg["model"]
        model = Restormer(
            in_channels=mcfg["in_channels"],
            out_channels=mcfg["out_channels"],
            dim=mcfg["dim"],
            num_blocks=mcfg["num_blocks"],
            num_refinement_blocks=mcfg["num_refinement_blocks"],
            heads=mcfg["heads"],
            ffn_expansion_factor=mcfg["ffn_expansion_factor"],
            bias=mcfg["bias"],
        )
    else:
        model = Restormer()

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer(
    model: Restormer,
    image: np.ndarray,
    mask: np.ndarray | None,
    device: torch.device,
    tile: bool = False,
    tile_size: int = 256,
    overlap: int = 32,
) -> np.ndarray:
    """
    Run inference on a single image.

    Args:
        model:     Loaded Restormer model.
        image:     [H, W, 3] float32 array in [0, 1], channels (Green, Red, NIR).
        mask:      Optional [H, W] binary cloud mask.
        device:    Compute device.
        tile:      Use tiled inference for large images.
        tile_size: Tile size (default 256).
        overlap:   Overlap between tiles.

    Returns:
        [H, W, 3] restored float32 array in [0, 1].
    """
    if tile:
        from ai.restormer.tiled_inference import tiled_infer
        return tiled_infer(model, image, mask, device, tile_size=tile_size, overlap=overlap)

    H, W, C = image.shape
    t = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)  # [1, 3, H, W]
    out = model(t)                                                             # [1, 3, H, W]
    out_np = out.squeeze(0).cpu().numpy().transpose(1, 2, 0)                  # [H, W, 3]

    # Apply mask: clear pixels keep original values
    if mask is not None:
        mask_3 = np.stack([mask, mask, mask], axis=-1)
        out_np = image * (1 - mask_3) + out_np * mask_3

    return np.clip(out_np, 0.0, 1.0).astype(np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restormer single-image inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--input", required=True, help="Path to cloudy image (.npy [H,W,3] or .tif GeoTIFF)")
    parser.add_argument("--output", required=True, help="Output path (.npy or .tif — matched to input format)")
    parser.add_argument("--mask", default=None, help="Path to binary cloud mask (.npy [H,W] or .tif)")
    parser.add_argument("--config", default=None, help="Optional train_config.yaml")
    parser.add_argument("--tile", action="store_true", help="Use tiled inference")
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.checkpoint, args.config, device)

    in_suffix = Path(args.input).suffix.lower()
    tiff_meta = None

    if in_suffix in {".tif", ".tiff"}:
        from ai.geospatial.tiff_io import read_tiff, write_tiff
        arr_chw, tiff_meta = read_tiff(args.input, normalize=True)
        image = arr_chw.transpose(1, 2, 0)  # CHW → HWC [H,W,3]
    else:
        image = np.load(args.input).astype(np.float32)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        # Normalize CHW → HWC if needed
        if image.ndim == 3 and image.shape[0] <= 13 and image.shape[2] > 13:
            image = image.transpose(1, 2, 0)

    mask: np.ndarray | None = None
    if args.mask:
        mask_suffix = Path(args.mask).suffix.lower()
        if mask_suffix in {".tif", ".tiff"}:
            from ai.geospatial.tiff_io import read_tiff
            mask_arr, _ = read_tiff(args.mask, normalize=False)
            mask = (mask_arr[0] > 0.5).astype(np.float32)
        else:
            mask = (np.load(args.mask) > 0.5).astype(np.float32)

    result = infer(
        model, image, mask, device,
        tile=args.tile,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_suffix = out_path.suffix.lower()
    if out_suffix in {".tif", ".tiff"} and tiff_meta is not None:
        from ai.geospatial.tiff_io import write_tiff
        write_tiff(str(out_path), result.transpose(2, 0, 1), tiff_meta)
    elif out_suffix in {".tif", ".tiff"}:
        # No source metadata — write a plain GeoTIFF without CRS
        from ai.geospatial.tiff_io import TiffMeta, write_tiff
        h, w, _ = result.shape
        meta = TiffMeta(width=w, height=h, count=3)
        write_tiff(str(out_path), result.transpose(2, 0, 1), meta)
    else:
        np.save(str(out_path), result)

    print(f"Saved reconstruction → {out_path}")


if __name__ == "__main__":
    main()
