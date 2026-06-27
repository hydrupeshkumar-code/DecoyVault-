"""
Batch LISS-IV Cloud Removal — production inference pipeline.

Reads real LISS-IV scenes, runs Restormer cloud removal, writes cloud-free
GeoTIFFs preserving CRS / geotransform so outputs are ready for NDVI, NDWI,
land-cover analysis, and any other downstream product.

Input formats supported:
  1. Bhoonidhi per-band TIFFs  — scene subdirectory containing BAND2.tif,
                                  BAND3.tif, BAND4.tif (auto-detected).
  2. Multi-band GeoTIFF        — single .tif with ≥3 bands [G, R, NIR].
  3. Pre-chipped .npy tiles    — [H, W, 3] float32 [0,1] chips from
                                  chip_liss4.py.

Usage:
    python -m ai.batch_infer \\
        --checkpoint  ai/restormer/checkpoints/phase2_final.pt \\
        --input-dir   raw/liss4/cloudy \\
        --output-dir  outputs/cloud_free \\
        [--mask-dir   raw/liss4/masks] \\
        [--config     ai/restormer/train_config.yaml] \\
        [--tile-size  512] [--overlap 64]

Output layout:
    outputs/cloud_free/
        <scene_id>.tif           — cloud-free GeoTIFF, float32 [0,1], same
                                   CRS / transform as input
        batch_report.json        — per-scene quality metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.restormer.infer import load_model, infer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Bhoonidhi band file names → LISS-IV G/R/NIR order
_LISS4_BANDS = ["BAND2", "BAND3", "BAND4"]
_TIFF_EXTS   = {".tif", ".tiff", ".TIF", ".TIFF"}


# ---------------------------------------------------------------------------
# Scene loading
# ---------------------------------------------------------------------------

def _is_bhoonidhi_dir(d: Path) -> bool:
    """True when directory has BAND2/3/4 TIFFs (Bhoonidhi delivery format)."""
    if not d.is_dir():
        return False
    return all(
        any((d / f"{b}{ext}").exists() for ext in _TIFF_EXTS)
        for b in _LISS4_BANDS
    )


def _load_bhoonidhi_scene(scene_dir: Path) -> tuple[np.ndarray, object]:
    """
    Stack BAND2/3/4 single-band TIFFs → [H, W, 3] float32 [0,1].
    Returns (image_hwc, tiff_meta) — meta is taken from BAND2 (Green).
    """
    from ai.geospatial.tiff_io import read_tiff, TiffMeta
    import rasterio

    bands = []
    meta_ref = None
    for band_name in _LISS4_BANDS:
        band_path = next(
            (scene_dir / f"{band_name}{ext}" for ext in _TIFF_EXTS
             if (scene_dir / f"{band_name}{ext}").exists()),
            None,
        )
        if band_path is None:
            raise FileNotFoundError(f"Missing {band_name} in {scene_dir}")
        arr, m = read_tiff(str(band_path), normalize=True)   # [1, H, W]
        bands.append(arr[0])
        if meta_ref is None:
            meta_ref = m

    stacked = np.stack(bands, axis=0)  # [3, H, W]
    if meta_ref is not None:
        meta_ref.count = 3
        meta_ref.band_descriptions = ["Green", "Red", "NIR"]
    return stacked.transpose(1, 2, 0), meta_ref   # [H, W, 3], TiffMeta


def _load_multiband_tiff(path: Path) -> tuple[np.ndarray, object]:
    """Load a ≥3-band GeoTIFF → [H, W, 3] float32 [0,1]."""
    from ai.geospatial.tiff_io import read_tiff
    arr, meta = read_tiff(str(path), band_indices=[0, 1, 2], normalize=True)
    return arr.transpose(1, 2, 0), meta   # [H, W, 3]


def _load_npy_chip(path: Path) -> tuple[np.ndarray, None]:
    """Load a .npy chip → [H, W, 3] float32 [0,1]."""
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] <= 4:
        arr = arr.transpose(1, 2, 0)   # CHW → HWC
    return arr, None


def load_scene(path: Path) -> tuple[np.ndarray, object]:
    """
    Auto-detect format and return ([H, W, 3] float32 [0,1], TiffMeta or None).
    """
    if _is_bhoonidhi_dir(path):
        return _load_bhoonidhi_scene(path)
    if path.suffix.lower() in _TIFF_EXTS:
        return _load_multiband_tiff(path)
    if path.suffix.lower() == ".npy":
        return _load_npy_chip(path)
    raise ValueError(f"Cannot load scene from {path} — unrecognised format.")


def _load_mask(mask_path: Path) -> np.ndarray:
    """Load a cloud mask → [H, W] float32 binary."""
    if mask_path.suffix.lower() in _TIFF_EXTS:
        from ai.geospatial.tiff_io import read_tiff
        arr, _ = read_tiff(str(mask_path), normalize=False)
        return (arr[0] > 0.5).astype(np.float32)
    m = np.load(mask_path).astype(np.float32)
    if m.ndim == 3:
        m = m[0]
    return (m > 0.5).astype(np.float32)


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def save_result(
    out_path: Path,
    result_hwc: np.ndarray,
    meta: object,
) -> None:
    """Write cloud-free result as GeoTIFF (or .npy if no meta available)."""
    result_chw = result_hwc.transpose(2, 0, 1)   # [3, H, W]

    if meta is not None:
        from ai.geospatial.tiff_io import write_tiff
        out_tif = out_path.with_suffix(".tif")
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        write_tiff(str(out_tif), result_chw, meta)
        log.info("  → %s", out_tif)
    else:
        out_npy = out_path.with_suffix(".npy")
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_npy), result_hwc)
        log.info("  → %s  (no GeoTIFF meta — saved as .npy)", out_npy)


# ---------------------------------------------------------------------------
# Per-scene metrics
# ---------------------------------------------------------------------------

def _scene_metrics(cloudy: np.ndarray, result: np.ndarray, mask: np.ndarray | None) -> dict:
    """Quick radiometric quality check: per-band mean shift and NDVI change."""
    metrics: dict = {}
    for c, name in enumerate(["green", "red", "nir"]):
        metrics[f"{name}_input_mean"]  = float(cloudy[..., c].mean())
        metrics[f"{name}_output_mean"] = float(result[..., c].mean())

    R_c, N_c = cloudy[..., 1], cloudy[..., 2]
    R_r, N_r = result[..., 1], result[..., 2]
    metrics["ndvi_input"]  = float(((N_c - R_c) / (N_c + R_c + 1e-8)).mean())
    metrics["ndvi_output"] = float(((N_r - R_r) / (N_r + R_r + 1e-8)).mean())
    metrics["ndvi_delta"]  = metrics["ndvi_output"] - metrics["ndvi_input"]

    if mask is not None:
        metrics["cloud_fraction"] = float(mask.mean())

    return metrics


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

def _discover_scenes(input_dir: Path) -> list[Path]:
    """
    Return all scenes (Bhoonidhi dirs, multi-band TIFFs, or .npy chips)
    found directly in input_dir.
    """
    scenes: list[Path] = []

    # Bhoonidhi subdirectories
    for d in sorted(input_dir.iterdir()):
        if _is_bhoonidhi_dir(d):
            scenes.append(d)

    # Multi-band GeoTIFFs at root level
    for f in sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff")):
        scenes.append(f)

    # Pre-chipped .npy files
    for f in sorted(input_dir.glob("*.npy")):
        scenes.append(f)

    return scenes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch LISS-IV cloud removal — produces cloud-free GeoTIFFs"
    )
    parser.add_argument("--checkpoint",  required=True, help="Model checkpoint (.pt)")
    parser.add_argument("--input-dir",   required=True, help="Directory of LISS-IV scenes")
    parser.add_argument("--output-dir",  required=True, help="Where to write cloud-free outputs")
    parser.add_argument("--mask-dir",    default=None,  help="Optional cloud mask directory")
    parser.add_argument("--config",      default=None,  help="train_config.yaml (for model arch)")
    parser.add_argument("--tile-size",   type=int, default=512,
                        help="Inference tile size in pixels (default 512)")
    parser.add_argument("--overlap",     type=int, default=64,
                        help="Tile overlap in pixels (default 64)")
    parser.add_argument("--hist-match",  action="store_true", default=True,
                        help="Rescale output brightness to match cloud-free input areas")
    parser.add_argument("--no-hist-match", dest="hist_match", action="store_false")
    parser.add_argument("--report",      default=None,
                        help="Path for JSON quality report (default: output_dir/batch_report.json)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Load model
    log.info("Loading model from %s …", args.checkpoint)
    model = load_model(args.checkpoint, args.config, device)

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    mask_dir   = Path(args.mask_dir) if args.mask_dir else None

    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = _discover_scenes(input_dir)
    if not scenes:
        log.error("No scenes found in %s", input_dir)
        sys.exit(1)
    log.info("Found %d scene(s) in %s", len(scenes), input_dir)

    report: list[dict] = []
    n_ok = n_err = 0

    for scene_path in scenes:
        scene_id = scene_path.stem if scene_path.is_file() else scene_path.name
        log.info("[%s] Processing …", scene_id)
        t0 = time.time()

        try:
            # Load cloudy scene
            cloudy_hwc, tiff_meta = load_scene(scene_path)
            H, W, _ = cloudy_hwc.shape
            log.info("  Size: %d × %d px", H, W)

            # Load cloud mask if available
            mask_hw: np.ndarray | None = None
            if mask_dir is not None:
                for ext in (".npy", ".tif", ".tiff"):
                    mp = mask_dir / f"{scene_id}{ext}"
                    if mp.exists():
                        mask_hw = _load_mask(mp)
                        break

            # Run tiled inference
            use_tile = (H > args.tile_size or W > args.tile_size)
            result_hwc = infer(
                model, cloudy_hwc, mask_hw, device,
                tile=use_tile,
                tile_size=args.tile_size,
                overlap=args.overlap,
                hist_match=args.hist_match,
            )

            # Save output
            save_result(output_dir / scene_id, result_hwc, tiff_meta)

            # Metrics
            m = _scene_metrics(cloudy_hwc, result_hwc, mask_hw)
            m["scene"] = scene_id
            m["elapsed_s"] = round(time.time() - t0, 1)
            m["size_px"] = f"{H}×{W}"
            report.append(m)

            log.info(
                "  NDVI: input=%+.3f → output=%+.3f (Δ%+.3f)  [%.1f s]",
                m["ndvi_input"], m["ndvi_output"], m["ndvi_delta"], m["elapsed_s"],
            )
            n_ok += 1

        except Exception as exc:
            log.error("[%s] Failed: %s", scene_id, exc)
            report.append({"scene": scene_id, "error": str(exc)})
            n_err += 1

    # Write report
    report_path = Path(args.report) if args.report else output_dir / "batch_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report → %s", report_path)
    log.info("Done — %d succeeded, %d failed.", n_ok, n_err)


if __name__ == "__main__":
    main()
