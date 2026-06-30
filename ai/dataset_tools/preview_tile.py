"""
Tile preview tool — visual sanity-check for cloud-removal training data.

WHY THIS EXISTS
---------------
SEN12MS-CR tiles are 13-band uint16 GeoTIFFs.  Windows Explorer, QGIS with
default settings, and most quick viewers show either the wrong bands (B1/B2/B3
are near-zero) or apply no contrast stretch — both produce black images.
The science data is correct; the viewer is broken.

This tool reads any raw .tif or prepared .npy tile, applies a per-band
2–98 % percentile stretch (identical to generate_demo.py), and writes a
viewable PNG.  In batch mode it dumps N random side-by-side cloudy|clear
pairs so you can eyeball the full dataset in one pass.

DISPLAY MODES
-------------
false_color (default):  NIR→R, Red→G, Green→B
    Standard remote-sensing false colour.  Healthy vegetation = vivid red.
    Clouds = white or light cyan.  Best for cloud-quality inspection.
natural:                Green→R, Red→G, NIR→B
    Closer to what the eye expects; makes clouds look white over green land.

USAGE
-----
Single tile (.tif or .npy):
    python -m ai.dataset_tools.preview_tile --input path/to/tile.tif

Prepared dataset pair (shows cloudy + clear side-by-side):
    python -m ai.dataset_tools.preview_tile --input datasets/sen12ms_cr/cloudy/pair_000001.npy

Batch — 20 random pairs from a prepared dataset:
    python -m ai.dataset_tools.preview_tile \\
        --input   datasets/sen12ms_cr \\
        --batch   20 \\
        --output  previews/

Batch from raw SEN12MS-CR .tif directory (single-side, no clear reference):
    python -m ai.dataset_tools.preview_tile \\
        --input    datasets/raw/SEN12MSCR/s2_cloudy \\
        --batch    20 \\
        --output   previews/ \\
        --s2-bands 2,3,7

Flags:
    --s2-bands  Comma-separated 0-indexed band selection for .tif files
                (default: 2,3,7  i.e. B3=Green, B4=Red, B8=NIR for S2)
    --scale     Divisor applied to raw .tif DN before stretching
                (default: 10000 for Sentinel-2; use 0 to skip)
    --display   false_color | natural  (default: false_color)
    --no-mask   Skip mask overlay even when masks/ folder is present
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stretching / colour-mapping helpers
# ---------------------------------------------------------------------------

_S2_DEFAULT_BANDS = [2, 3, 7]  # B3=Green, B4=Red, B8=NIR (0-indexed)


def _percentile_stretch(arr: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """Per-band 2–98 % percentile stretch to [0, 1], ignoring NaN.

    Each band is stretched independently so all three carry information even
    when one band has very different absolute values (e.g. NIR >> visible).
    """
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[0]):
        band = arr[c].astype(np.float32)
        valid = band[~np.isnan(band)]
        if valid.size == 0:
            out[c] = 0.0
            continue
        lo = float(np.percentile(valid, lo_pct))
        hi = float(np.percentile(valid, hi_pct))
        if hi <= lo:
            out[c] = 0.0
        else:
            out[c] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return out


def _to_uint8(arr_01: np.ndarray) -> np.ndarray:
    """[C, H, W] float32 [0,1] → [H, W, 3] uint8 for PIL."""
    arr = np.nan_to_num(arr_01, nan=0.0)
    return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(1, 2, 0)


def _channel_order(arr_grn: np.ndarray, display: str) -> np.ndarray:
    """
    Reorder [G, R, N] → display RGB.

    false_color: NIR→R, Red→G, Green→B  (standard remote sensing)
    natural:     Green→R, Red→G, NIR→B  (hinted natural colour)
    """
    G, R, N = arr_grn[0], arr_grn[1], arr_grn[2]
    if display == "false_color":
        return np.stack([N, R, G], axis=0)
    if display == "natural":
        return np.stack([G, R, N], axis=0)
    raise ValueError(f"Unknown display mode: {display!r}. Use 'false_color' or 'natural'.")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_tif(path: Path, bands: list[int], scale: float) -> np.ndarray:
    """Read a GeoTIFF, select bands, apply DN→[0,1] pre-scale if scale>0.

    Returns [C, H, W] float32 (un-stretched; stretching is done separately
    so the caller can choose how to combine multi-tile statistics).
    """
    try:
        import rasterio
    except ImportError as e:
        raise ImportError("pip install rasterio  — required for .tif support") from e

    with rasterio.open(str(path)) as src:
        # rasterio bands are 1-indexed
        rb = [b + 1 for b in bands]
        if any(b > src.count for b in rb):
            raise ValueError(
                f"{path.name}: requested bands {bands} but file has only {src.count} bands."
            )
        data = src.read(rb).astype(np.float32)  # [C, H, W]
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    if scale > 0:
        data = data / scale
    return data


def _load_npy(path: Path) -> np.ndarray:
    """Load a prepared .npy tile → [C, H, W] float32.

    Prepared tiles are [H, W, C] on disk (as written by prepare_dataset.py),
    so we transpose.  Raw 1-D or 2-D .npy are handled gracefully.
    """
    arr = np.load(str(path)).astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] <= 16:
        arr = arr.transpose(2, 0, 1)  # [H,W,C] → [C,H,W]
    elif arr.ndim == 2:
        arr = arr[np.newaxis]         # [H,W] → [1,H,W]
    return arr


def _load_tile(path: Path, bands: list[int], scale: float) -> np.ndarray:
    """Unified loader for .tif/.tiff or .npy → [C, H, W] float32."""
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        return _load_tif(path, bands, scale)
    if suffix == ".npy":
        arr = _load_npy(path)
        if arr.shape[0] > 3:
            # Multi-band .npy (unusual; would be raw): select bands
            arr = arr[bands]
        return arr
    raise ValueError(f"Unsupported format: {suffix}. Use .tif or .npy.")


# ---------------------------------------------------------------------------
# Single-tile PNG writer
# ---------------------------------------------------------------------------

def _make_preview(
    arr: np.ndarray,
    display: str,
    label: Optional[str] = None,
    mask: Optional[np.ndarray] = None,
) -> "PIL.Image.Image":
    """Build a preview PIL Image from a [C, H, W] float32 array.

    If a binary mask [H, W] is supplied, cloud pixels are tinted magenta.
    """
    from PIL import Image, ImageDraw, ImageFont

    stretched = _percentile_stretch(arr)          # [C, H, W] in [0,1]
    rgb_arr   = _channel_order(stretched, display)  # reorder → display
    rgb_uint8 = _to_uint8(rgb_arr)                # [H, W, 3] uint8

    img = Image.fromarray(rgb_uint8, mode="RGB")

    # Mask overlay (cloud pixels → semi-transparent magenta)
    if mask is not None:
        m = (mask > 0.5).astype(np.uint8) * 120   # alpha channel
        tint = Image.new("RGB", img.size, color=(255, 0, 180))
        alpha = Image.fromarray(m, mode="L")
        img = Image.composite(tint, img, alpha)

    # Label banner at top
    if label:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        draw.rectangle([0, 0, img.width, 18], fill=(0, 0, 0))
        draw.text((4, 2), label, fill=(255, 255, 0), font=font)

    return img


def preview_single(
    path: Path,
    output: Optional[Path],
    bands: list[int],
    scale: float,
    display: str,
    show_mask: bool = True,
) -> Path:
    """Write a single preview PNG.  Returns the output path."""
    from PIL import Image

    arr  = _load_tile(path, bands, scale)
    mask = None

    if show_mask and path.suffix.lower() == ".npy":
        # Look for a sibling mask in masks/ (prepared dataset layout)
        mask_path = path.parent.parent / "masks" / path.name
        if mask_path.exists():
            m = _load_npy(mask_path)
            mask = m[0] if m.ndim == 3 else m

    img = _make_preview(arr, display, label=path.name, mask=mask)

    if output is None:
        out_path = path.with_suffix(".preview.png")
    else:
        output.mkdir(parents=True, exist_ok=True)
        out_path = output / (path.stem + ".preview.png")

    img.save(str(out_path))
    log.info("Wrote %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Batch mode (prepared dataset: cloudy + clear side-by-side)
# ---------------------------------------------------------------------------

def _discover_tiles(root: Path) -> list[Path]:
    """Find all .npy and .tif files under root, sorted."""
    exts = {".npy", ".tif", ".tiff"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file())


def preview_batch(
    dataset_dir: Path,
    output: Path,
    n: int,
    bands: list[int],
    scale: float,
    display: str,
    show_mask: bool = True,
    seed: int = 42,
) -> int:
    """
    Write N side-by-side cloudy|clear preview PNGs from a prepared dataset.

    Expects dataset_dir/{cloudy,clear,masks}/*.npy (the output layout of
    prepare_dataset.py).  Falls back to single-directory batch when no
    cloudy/ subdirectory exists.

    Returns the number of previews written.
    """
    from PIL import Image

    cloudy_dir = dataset_dir / "cloudy"
    clear_dir  = dataset_dir / "clear"
    masks_dir  = dataset_dir / "masks"

    if cloudy_dir.is_dir():
        stems = sorted({p.stem for p in cloudy_dir.glob("*.npy")})
        if not stems:
            stems = sorted({p.stem for p in cloudy_dir.glob("*.tif*")})
    else:
        # Raw directory: just single-side previews
        log.info("No cloudy/ subdir found — falling back to single-side previews.")
        tiles = _discover_tiles(dataset_dir)
        rng = random.Random(seed)
        rng.shuffle(tiles)
        tiles = tiles[:n]
        written = 0
        for t in tiles:
            try:
                preview_single(t, output, bands, scale, display, show_mask=False)
                written += 1
            except Exception as e:
                log.warning("Skipping %s: %s", t.name, e)
        return written

    rng = random.Random(seed)
    sample = rng.sample(stems, min(n, len(stems)))
    output.mkdir(parents=True, exist_ok=True)
    written = 0

    for stem in sample:
        ext = ".npy"
        c_path = cloudy_dir / f"{stem}{ext}"
        cl_path = clear_dir / f"{stem}{ext}" if clear_dir.is_dir() else None
        m_path  = masks_dir / f"{stem}{ext}" if (masks_dir.is_dir() and show_mask) else None

        if not c_path.exists():
            log.warning("Cloudy tile not found: %s", c_path)
            continue

        try:
            c_arr = _load_tile(c_path, bands, scale)
            c_img = _make_preview(
                c_arr, display,
                label=f"{stem} — CLOUDY",
                mask=_load_npy(m_path)[0] if (m_path and m_path.exists()) else None,
            )
        except Exception as e:
            log.warning("Skipping %s (cloudy): %s", stem, e)
            continue

        panels = [c_img]

        if cl_path and cl_path.exists():
            try:
                cl_arr = _load_tile(cl_path, bands, scale)
                cl_img = _make_preview(cl_arr, display, label=f"{stem} — CLEAR")
                panels.append(cl_img)
            except Exception as e:
                log.warning("Skipping %s (clear): %s", stem, e)

        # Stitch panels horizontally with a 2-px white separator
        H = max(p.height for p in panels)
        W = sum(p.width for p in panels) + 2 * (len(panels) - 1)
        canvas = Image.new("RGB", (W, H), color=(255, 255, 255))
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width + 2

        out_path = output / f"{stem}.png"
        canvas.save(str(out_path))
        log.info("Wrote %s", out_path)
        written += 1

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_bands(s: Optional[str]) -> list[int]:
    if not s:
        return _S2_DEFAULT_BANDS
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(
        description="Write viewable PNG previews of raw .tif or prepared .npy satellite tiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input",  required=True, help="Path to a single tile or a dataset directory")
    p.add_argument("--output", default=None,  help="Output directory (default: beside input file)")
    p.add_argument("--batch",  type=int, default=0,
                   help="Number of random previews to write (0 = single-tile mode)")
    p.add_argument("--s2-bands", default=None,
                   help="Comma-separated 0-indexed band indices for .tif files "
                        "(default: 2,3,7 = B3/B4/B8 for Sentinel-2)")
    p.add_argument("--scale", type=float, default=10_000.0,
                   help="Pre-stretch DN divisor for raw .tif files (default 10000 for S2; 0 to skip)")
    p.add_argument("--display", choices=["false_color", "natural"], default="false_color",
                   help="Colour composite (default: false_color = NIR→R, Red→G, Green→B)")
    p.add_argument("--no-mask", action="store_true",
                   help="Skip cloud-mask overlay even when a masks/ folder is present")
    p.add_argument("--seed", type=int, default=42, help="Random seed for batch sampling")
    args = p.parse_args()

    inp    = Path(args.input)
    out    = Path(args.output) if args.output else None
    bands  = _parse_bands(args.s2_bands)
    scale  = args.scale
    display = args.display
    show_mask = not args.no_mask

    try:
        from PIL import Image as _  # noqa: F401
    except ImportError:
        raise SystemExit("Pillow is required for PNG output.  pip install Pillow")

    if args.batch > 0:
        if not inp.is_dir():
            raise SystemExit(f"--batch requires --input to be a directory, got: {inp}")
        written = preview_batch(inp, out or inp / "previews", args.batch, bands, scale, display, show_mask, args.seed)
        print(f"\nWrote {written} preview(s) → {out or inp / 'previews'}")
    else:
        if inp.is_dir():
            raise SystemExit("--input is a directory but --batch was not given. Add --batch N.")
        out_path = preview_single(inp, out, bands, scale, display, show_mask)
        print(f"\nWrote preview → {out_path}")

    mode_note = "(NIR→R, Red→G, Green→B)" if display == "false_color" else "(Green→R, Red→G, NIR→B)"
    print(f"Display mode: {display} {mode_note}")
    print("Healthy vegetation = vivid red in false_color mode; clouds = white/cyan.")


if __name__ == "__main__":
    main()
