"""
LISS-IV Cloud Removal Demo — inference + visualization.

Scans the chip pool to find a chip with partial clouds (good for demo),
runs the Restormer model, and saves a 3-panel comparison PNG with proper
percentile contrast stretching so results are visually interpretable.

Usage:
    python -m ai.demo_liss4 \\
        --checkpoint ai/restormer/checkpoints/phase2_final.pt \\
        --chips     datasets/liss4 \\
        [--output   demo_result.png] \\
        [--config   ai/restormer/train_config.yaml] \\
        [--chip     chip012]        # force a specific chip stem
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.restormer.infer import load_model, infer


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def percentile_stretch(img_hwc: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """
    Apply per-channel percentile stretch so results are visually interpretable
    regardless of absolute reflectance levels or domain shift.
    """
    out = np.empty_like(img_hwc, dtype=np.float32)
    for c in range(img_hwc.shape[2]):
        ch = img_hwc[:, :, c]
        p_lo = np.percentile(ch, lo)
        p_hi = np.percentile(ch, hi)
        out[:, :, c] = np.clip((ch - p_lo) / (p_hi - p_lo + 1e-8), 0.0, 1.0)
    return out


def to_false_colour_uint8(img_hwc: np.ndarray, stretch: bool = True) -> np.ndarray:
    """
    Convert [H,W,3] (Green=0, Red=1, NIR=2) to CIR false-colour uint8.

    CIR mapping: R ← NIR, G ← Red, B ← Green  (vegetation appears red).
    Percentile stretch is applied by default so output is always visible.
    """
    if stretch:
        img_hwc = percentile_stretch(img_hwc)
    nir  = img_hwc[:, :, 2]
    red  = img_hwc[:, :, 1]
    green = img_hwc[:, :, 0]
    cir = np.stack([nir, red, green], axis=-1)
    return (np.clip(cir, 0, 1) * 255).astype(np.uint8)


def make_diff_map_uint8(cloudy_hwc: np.ndarray, result_hwc: np.ndarray) -> np.ndarray:
    """
    Absolute difference map, averaged across bands, shown as a heat-map.
    Red = large change (cloud removed), dark = unchanged (cloud-free ground preserved).
    """
    diff = np.abs(result_hwc - cloudy_hwc).mean(axis=2)  # [H, W]
    p98 = np.percentile(diff, 98)
    diff_norm = np.clip(diff / (p98 + 1e-8), 0, 1)
    # Colourmap: low=dark blue, high=bright red using a simple 3-stop gradient
    r = diff_norm
    g = np.zeros_like(r)
    b = 1.0 - diff_norm
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def save_comparison_png(
    cloudy_hwc: np.ndarray,
    result_hwc: np.ndarray,
    mask_hw: np.ndarray | None,
    out_path: str | Path,
    title: str = "",
) -> None:
    """Save a 3-panel (or 4-panel with mask) comparison PNG."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("[demo] Pillow not installed — saving raw .npy arrays instead.")
        np.save(str(out_path).replace(".png", "_cloudy.npy"),  cloudy_hwc)
        np.save(str(out_path).replace(".png", "_result.npy"),  result_hwc)
        return

    panels = [
        ("Cloudy (CIR)",         to_false_colour_uint8(cloudy_hwc)),
        ("Cloud-Removed (CIR)",  to_false_colour_uint8(result_hwc)),
        ("Change Map",           make_diff_map_uint8(cloudy_hwc, result_hwc)),
    ]
    if mask_hw is not None:
        mask_u8 = (mask_hw * 255).astype(np.uint8)
        panels.append(("Cloud Mask", np.stack([mask_u8, mask_u8, mask_u8], axis=-1)))

    H, W = cloudy_hwc.shape[:2]
    label_h = 32
    n = len(panels)
    canvas = np.zeros((H + label_h, W * n, 3), dtype=np.uint8)

    for i, (label, arr) in enumerate(panels):
        canvas[label_h:, W * i: W * (i + 1), :] = arr

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    for i, (label, _) in enumerate(panels):
        draw.text((W * i + 8, 6), label, fill=(255, 255, 220), font=font)

    if title:
        draw.text((8, label_h + H - 28), title, fill=(255, 255, 200), font=font)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    print(f"[demo] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Chip scanning
# ---------------------------------------------------------------------------

def scan_chips(
    cloudy_dir: Path,
    mask_dir: Path | None,
    target_cloud_lo: float = 0.15,
    target_cloud_hi: float = 0.60,
    min_ndvi: float = 0.0,
    max_scan: int = 100,
) -> list[tuple[str, float, float]]:
    """
    Return (stem, cloud_fraction, clear_ndvi) triples with cloud fraction in [lo, hi]
    and clear-pixel NDVI >= min_ndvi, sorted by clear NDVI descending so the most
    vegetation-rich (rice/agriculture) chips come first.

    Band order in chips: [0]=Green, [1]=Red, [2]=NIR  (LISS-IV convention).
    """
    candidates = []
    for p in sorted(cloudy_dir.glob("*.npy"))[:max_scan]:
        arr = np.load(p).astype(np.float32)
        if arr.ndim == 3 and arr.shape[2] <= 4:
            arr = arr.transpose(2, 0, 1)  # HWC → CHW

        if mask_dir is not None and (mask_dir / p.name).exists():
            mask = np.load(mask_dir / p.name)
            if mask.ndim == 3:
                mask = mask[0]
            cf = float(mask.mean())
            clear_px = mask < 0.5        # boolean [H,W]
        else:
            # Estimate cloud mask: bright pixels (all bands > 0.6) ≈ cloud
            brightness = arr.mean(axis=0)
            clear_px = brightness < 0.6
            cf = float((~clear_px).mean())

        if not (target_cloud_lo <= cf <= target_cloud_hi):
            continue

        # Compute NDVI over clear-sky pixels only
        red = arr[1][clear_px]
        nir = arr[2][clear_px]
        if red.size > 100:
            ndvi = float(((nir - red) / (nir + red + 1e-8)).mean())
        else:
            ndvi = 0.0

        if ndvi >= min_ndvi:
            candidates.append((p.stem, cf, ndvi))

    # Sort by NDVI descending — vegetation-rich chips first
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LISS-IV cloud removal demo")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--chips",      required=True, help="Root of liss4 chip dataset")
    parser.add_argument("--output",     default="demo_result.png")
    parser.add_argument("--config",     default=None)
    parser.add_argument("--chip",       default=None, help="Force a specific chip stem")
    parser.add_argument("--min-ndvi",   type=float, default=0.2,
                        help="Minimum clear-sky NDVI to qualify (0.2=vegetation, 0.4=dense rice)")
    parser.add_argument("--tile",       action="store_true")
    parser.add_argument("--tile-size",  type=int, default=256)
    parser.add_argument("--overlap",    type=int, default=32)
    parser.add_argument(
        "--hist-match", action="store_true", default=True,
        help="Rescale output statistics to match cloud-free input (fixes domain-shift brightness). On by default.",
    )
    parser.add_argument(
        "--no-hist-match", dest="hist_match", action="store_false",
        help="Disable histogram matching (show raw model output).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[demo] Device: {device}")

    chips_root = Path(args.chips)
    cloudy_dir = chips_root / "cloudy"
    # Accept both masks/ (LISS-IV convention) and mask/ (RICE2 convention)
    mask_dir = next(
        (chips_root / d for d in ("masks", "mask") if (chips_root / d).exists()),
        None,
    )

    # --- pick chip ---
    if args.chip:
        stem = args.chip
        cf   = None
        ndvi_clear = None
    else:
        print(f"[demo] Scanning chips for vegetation-rich partial-cloud example (min NDVI={args.min_ndvi}) …")
        candidates = scan_chips(cloudy_dir, mask_dir, min_ndvi=args.min_ndvi)
        if not candidates:
            print("[demo] No chips with NDVI >= {args.min_ndvi} found. Relaxing to any vegetation …")
            candidates = scan_chips(cloudy_dir, mask_dir, min_ndvi=0.0)
        if not candidates:
            print("[demo] No chips with cloud fraction 0.15-0.60 found. Trying full range …")
            candidates = scan_chips(cloudy_dir, mask_dir, 0.05, 0.95, min_ndvi=0.0)
        if not candidates:
            first = next(cloudy_dir.glob("*.npy"), None)
            if first is None:
                sys.exit(f"[demo] No .npy chips found in {cloudy_dir}")
            candidates = [(first.stem, -1.0, 0.0)]
        stem, cf, ndvi_clear = candidates[0]
        print(f"[demo] Selected chip: {stem}  cloud={cf:.2f}  clear-NDVI={ndvi_clear:+.3f}")

    # --- load chip ---
    chip_path = cloudy_dir / f"{stem}.npy"
    cloudy_arr = np.load(chip_path).astype(np.float32)
    if cloudy_arr.ndim == 3 and cloudy_arr.shape[0] <= 4:
        cloudy_hwc = cloudy_arr.transpose(1, 2, 0)  # CHW → HWC
    else:
        cloudy_hwc = cloudy_arr

    mask_hw: np.ndarray | None = None
    if mask_dir is not None:
        mask_path = mask_dir / f"{stem}.npy"
        if mask_path.exists():
            m = np.load(mask_path).astype(np.float32)
            mask_hw = m[0] if m.ndim == 3 else m

    # --- load model & run inference ---
    print(f"[demo] Loading model from {args.checkpoint} …")
    model = load_model(args.checkpoint, args.config, device)

    print(f"[demo] Running inference (hist_match={args.hist_match}) …")
    result_hwc = infer(
        model, cloudy_hwc, mask_hw, device,
        tile=args.tile, tile_size=args.tile_size, overlap=args.overlap,
        hist_match=args.hist_match,
    )

    # --- print stats ---
    def _stats(name: str, arr: np.ndarray) -> None:
        print(f"[demo] {name:12s}  min={arr.min():.4f}  mean={arr.mean():.4f}  max={arr.max():.4f}")

    _stats("Cloudy", cloudy_hwc)
    _stats("Result", result_hwc)
    if mask_hw is not None:
        print(f"[demo] Cloud frac = {mask_hw.mean():.3f}")

    # Band ratios
    G_c, R_c, N_c = cloudy_hwc[..., 0], cloudy_hwc[..., 1], cloudy_hwc[..., 2]
    G_r, R_r, N_r = result_hwc[..., 0], result_hwc[..., 1], result_hwc[..., 2]
    ndvi_c = ((N_c - R_c) / (N_c + R_c + 1e-8)).mean()
    ndvi_r = ((N_r - R_r) / (N_r + R_r + 1e-8)).mean()
    print(f"[demo] NDVI  cloudy={ndvi_c:+.3f}  result={ndvi_r:+.3f}  Δ={ndvi_r - ndvi_c:+.3f}")

    # --- save comparison ---
    title_str = (
        f"Chip: {stem}  |  Cloud: {(cf or 0):.0%}"
        f"  |  Clear-NDVI (input): {ndvi_c:+.2f}"
        f"  |  NDVI (result): {ndvi_r:+.2f}"
    )
    save_comparison_png(cloudy_hwc, result_hwc, mask_hw, args.output, title=title_str)

    # Also save raw result .npy
    result_path = Path(args.output).with_suffix(".npy")
    np.save(str(result_path), result_hwc)
    print(f"[demo] Raw result array → {result_path}")


if __name__ == "__main__":
    main()
