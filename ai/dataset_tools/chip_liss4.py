"""
Chip large LISS-IV GeoTIFF scenes into small training tiles.

WHY THIS EXISTS
---------------
A full LISS-IV scene is ~70 km × 70 km at 5.8 m resolution (~12 000 × 12 000 px,
~500 MB compressed). Loading one for a 128 × 128 training crop is 4 000× too
expensive. This tool slices each paired (cloudy, clear) scene into small chips
and saves them as .npy so the LISS4Dataset loader sees many cheap files.

Typical yield: 1 scene pair → ~550 chips (512 px stride, skip > 30 % nodata)

USAGE
-----
# Minimal: raw dir pair → ready dataset
python -m ai.dataset_tools.chip_liss4 \
    --cloudy-dir  raw/liss4/cloudy \
    --clear-dir   raw/liss4/clear \
    --output      datasets/liss4 \
    --chip-size   512 --stride 512

# With masks supplied + band selection for multi-band raw products
python -m ai.dataset_tools.chip_liss4 \
    --cloudy-dir  raw/liss4/cloudy \
    --clear-dir   raw/liss4/clear \
    --mask-dir    raw/liss4/masks \
    --output      datasets/liss4 \
    --bands 0,1,2 --chip-size 512 --stride 480

OUTPUT LAYOUT (consumed directly by LISS4Dataset)
--------------------------------------------------
datasets/liss4/
    cloudy/  scene001_chip000001.npy   [H, W, 3] float32 [0,1]
    clear/   scene001_chip000001.npy   [H, W, 3] float32 [0,1]
    masks/   scene001_chip000001.npy   [H, W]    float32 binary
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_TIFF_EXTS = {".tif", ".tiff", ".TIF", ".TIFF"}

# Bhoonidhi LISS-IV delivers bands as separate files inside a scene folder.
# Band mapping: BAND2=Green, BAND3=Red, BAND4=NIR (matches our G/R/NIR convention).
_LISS4_BAND_FILES = ["BAND2", "BAND3", "BAND4"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _read_tiff(path: Path) -> np.ndarray:
    """Load a GeoTIFF as [C, H, W] float32 (raw DN, un-normalised)."""
    from ai.geospatial.tiff_io import load_any
    arr, _ = load_any(path, normalize=False)
    return arr.astype(np.float32)


def _is_liss4_scene_dir(d: Path) -> bool:
    """True if directory contains Bhoonidhi-style BAND2/BAND3/BAND4 files."""
    if not d.is_dir():
        return False
    for band in _LISS4_BAND_FILES:
        if not any((d / f"{band}{ext}").exists() for ext in _TIFF_EXTS):
            return False
    return True


def _read_liss4_scene_dir(d: Path) -> np.ndarray:
    """
    Stack Bhoonidhi BAND2/BAND3/BAND4 single-band TIFFs into [3, H, W] float32.
    Band order: Green (BAND2), Red (BAND3), NIR (BAND4).
    """
    bands = []
    for band_name in _LISS4_BAND_FILES:
        band_path = None
        for ext in _TIFF_EXTS:
            p = d / f"{band_name}{ext}"
            if p.exists():
                band_path = p
                break
        if band_path is None:
            raise FileNotFoundError(f"{band_name} not found in {d}")
        arr = _read_tiff(band_path)
        # Each band file is [1, H, W] or [H, W]
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        elif arr.shape[0] != 1:
            arr = arr[0:1]
        bands.append(arr)
    return np.concatenate(bands, axis=0).astype(np.float32)  # [3, H, W]


def _discover_scenes(root: Path) -> list[tuple[str, Path]]:
    """
    Return (stem, source) pairs from root, supporting two layouts:
      1. Bhoonidhi: root/SceneFolder/{BAND2,BAND3,BAND4}.tif  (subdirs)
      2. Classic:   root/scene.tif                             (flat TIFFs)
    """
    scenes: list[tuple[str, Path]] = []
    for child in sorted(root.iterdir()):
        if _is_liss4_scene_dir(child):
            scenes.append((child.name, child))
        elif child.is_file() and child.suffix.lower() in _TIFF_EXTS:
            scenes.append((child.stem, child))
    return scenes


def _read_scene(source: Path) -> np.ndarray:
    """Load a scene (dir or file) as [3, H, W] float32."""
    if source.is_dir():
        return _read_liss4_scene_dir(source)
    return _read_tiff(source)


def _find_match(stem: str, directory: Path) -> Optional[Path]:
    for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Band selection + normalisation
# ---------------------------------------------------------------------------

def _select_bands(arr: np.ndarray, bands: Optional[list[int]]) -> np.ndarray:
    """Return [3, H, W] slice in Green / Red / NIR order."""
    C = arr.shape[0]
    if C == 3 and bands is None:
        return arr
    if bands is not None:
        if max(bands) >= C:
            raise ValueError(f"Band index {max(bands)} out of range for {C}-band image.")
        return arr[bands]
    if C >= 3:
        log.warning("No --bands given for %d-band file; using first 3.", C)
        return arr[:3]
    raise ValueError(f"Cannot produce 3 bands from {C}-band array.")


def _normalize_percentile(arr: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Per-band percentile stretch to [0, 1]."""
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[0]):
        band = arr[c]
        valid = band[band > 0]
        if valid.size == 0:
            out[c] = 0.0
            continue
        p_lo = float(np.percentile(valid, lo))
        p_hi = float(np.percentile(valid, hi))
        if p_hi <= p_lo:
            out[c] = 0.0
        else:
            out[c] = np.clip((band - p_lo) / (p_hi - p_lo), 0.0, 1.0)
    return out


def _normalize_dn(arr: np.ndarray, max_dn: float = 1023.0) -> np.ndarray:
    """Divide by max DN value (LISS-IV is 10-bit: 0–1023)."""
    return np.clip(arr / max_dn, 0.0, 1.0).astype(np.float32)


def _normalize(arr: np.ndarray, mode: str, max_dn: float) -> np.ndarray:
    if mode == "percentile":
        return _normalize_percentile(arr)
    if mode == "dn":
        return _normalize_dn(arr, max_dn)
    if mode == "auto":
        mx = float(arr.max())
        if mx <= 1.5:
            return arr.astype(np.float32)
        if mx <= 255.0:
            return np.clip(arr / 255.0, 0.0, 1.0).astype(np.float32)
        return _normalize_percentile(arr)  # safest for unknown DN range
    raise ValueError(f"Unknown normalize mode: {mode!r}")


# ---------------------------------------------------------------------------
# Mask derivation
# ---------------------------------------------------------------------------

def _derive_mask(cloudy: np.ndarray, clear: np.ndarray) -> np.ndarray:
    """[H, W] binary mask from normalised [3, H, W] pair (clouds = bright delta)."""
    diff = cloudy.mean(axis=0) - clear.mean(axis=0)
    bright = cloudy.mean(axis=0)
    mask = ((diff > 0.04) | (bright > 0.55)).astype(np.float32)
    try:
        from scipy import ndimage as ndi
        s = np.ones((3, 3), dtype=bool)
        mask = ndi.binary_closing(mask > 0.5, structure=s).astype(np.float32)
        mask = ndi.binary_opening(mask > 0.5, structure=s).astype(np.float32)
    except Exception:
        pass
    return mask


# ---------------------------------------------------------------------------
# Chipping
# ---------------------------------------------------------------------------

def _chip_array(arr: np.ndarray, chip: int, stride: int) -> list[tuple[int, int, np.ndarray]]:
    """Yield (row, col, chip_array) for a [C, H, W] array."""
    _, H, W = arr.shape
    chips = []
    for r in range(0, H - chip + 1, stride):
        for c in range(0, W - chip + 1, stride):
            chips.append((r, c, arr[:, r:r + chip, c:c + chip]))
    return chips


def _nodata_fraction(arr: np.ndarray) -> float:
    """Fraction of pixels that are exactly zero across all bands."""
    return float((arr.sum(axis=0) == 0).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def chip_scenes(
    cloudy_dir: Path,
    clear_dir: Path,
    output: Path,
    mask_dir: Optional[Path] = None,
    bands: Optional[list[int]] = None,
    normalize: str = "auto",
    max_dn: float = 1023.0,
    chip_size: int = 512,
    stride: int = 512,
    max_nodata: float = 0.30,
    min_cloud: float = 0.02,
    max_cloud: float = 0.95,
) -> dict:
    cloudy_scenes = _discover_scenes(cloudy_dir)
    if not cloudy_scenes:
        raise RuntimeError(
            f"No scenes found in {cloudy_dir}. "
            "Expected either subdirs with BAND2/BAND3/BAND4.tif (Bhoonidhi format) "
            "or flat .tif files."
        )

    # Build a lookup for clear scenes by stem
    clear_scenes = {stem: src for stem, src in _discover_scenes(clear_dir)}

    out_cloudy = output / "cloudy"
    out_clear  = output / "clear"
    out_masks  = output / "masks"
    for d in (out_cloudy, out_clear, out_masks):
        d.mkdir(parents=True, exist_ok=True)

    total_chips = 0
    skipped_nodata = 0
    skipped_cloud = 0
    scenes_processed = 0

    for stem, cloudy_source in cloudy_scenes:
        # Match clear scene by stem; for Bhoonidhi dirs the stems are the long
        # scene-folder names, so they must match exactly between cloudy/ and clear/.
        clear_source = clear_scenes.get(stem)
        if clear_source is None:
            log.warning("No clear match for '%s' — skipping.", stem)
            continue

        log.info("Processing scene: %s", stem)
        try:
            cloudy_raw = _read_scene(cloudy_source)
            clear_raw  = _read_scene(clear_source)
        except Exception as e:
            log.warning("Failed to read %s: %s", stem, e)
            continue

        # Bhoonidhi scenes already arrive in G/R/NIR order (BAND2/3/4);
        # only apply band selection for flat multi-band TIFFs.
        try:
            cloudy_3 = _select_bands(cloudy_raw, bands)
            clear_3  = _select_bands(clear_raw, bands)
        except ValueError as e:
            log.warning("Band selection failed for %s: %s", stem, e)
            continue

        if cloudy_3.shape != clear_3.shape:
            log.warning("Shape mismatch %s vs %s for %s — skipping.", cloudy_3.shape, clear_3.shape, stem)
            continue

        cloudy_n = _normalize(cloudy_3, normalize, max_dn)
        clear_n  = _normalize(clear_3, normalize, max_dn)

        if mask_dir is not None:
            mask_path = _find_match(stem, mask_dir)
            if mask_path is not None:
                try:
                    mask_raw, _ = __import__("ai.geospatial.tiff_io", fromlist=["load_any"]).load_any(mask_path, normalize=False)
                    mask_2d = mask_raw[0] if mask_raw.ndim == 3 else mask_raw
                    mask_2d = (mask_2d > 0.5).astype(np.float32)
                except Exception:
                    mask_2d = _derive_mask(cloudy_n, clear_n)
            else:
                mask_2d = _derive_mask(cloudy_n, clear_n)
        else:
            mask_2d = _derive_mask(cloudy_n, clear_n)

        mask_3 = mask_2d[np.newaxis]  # [1, H, W] for chipping alignment

        cloudy_chips = _chip_array(cloudy_n, chip_size, stride)
        clear_chips  = _chip_array(clear_n,  chip_size, stride)
        mask_chips   = _chip_array(mask_3,   chip_size, stride)

        scene_chips = 0
        for (r, c, cl_chip), (_, _, cr_chip), (_, _, mk_chip) in zip(cloudy_chips, clear_chips, mask_chips):
            mk_2d = mk_chip[0]
            nd = _nodata_fraction(cl_chip)
            cf = float(mk_2d.mean())

            if nd > max_nodata:
                skipped_nodata += 1
                continue
            if not (min_cloud <= cf <= max_cloud):
                skipped_cloud += 1
                continue

            safe_stem = stem[:40].replace(" ", "_")
            chip_name = f"{safe_stem}_chip{total_chips + 1:06d}"
            np.save(out_cloudy / f"{chip_name}.npy", cl_chip.transpose(1, 2, 0))
            np.save(out_clear  / f"{chip_name}.npy", cr_chip.transpose(1, 2, 0))
            np.save(out_masks  / f"{chip_name}.npy", mk_2d)

            total_chips  += 1
            scene_chips  += 1

        log.info("  → %d chips from scene %s", scene_chips, stem)
        scenes_processed += 1

    return {
        "scenes_processed": scenes_processed,
        "chips_written": total_chips,
        "skipped_nodata": skipped_nodata,
        "skipped_cloud_fraction": skipped_cloud,
        "output": str(output),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Chip large LISS-IV scenes into training tiles")
    p.add_argument("--cloudy-dir", required=True)
    p.add_argument("--clear-dir",  required=True)
    p.add_argument("--mask-dir",   default=None)
    p.add_argument("--output",     default="datasets/liss4")
    p.add_argument("--bands",      default=None,
                   help="0-indexed band order -> Green,Red,NIR  e.g. '0,1,2'")
    p.add_argument("--normalize",  default="auto",
                   choices=["auto", "percentile", "dn"],
                   help="auto=smart detect, percentile=2-98%% stretch, dn=divide by max-dn")
    p.add_argument("--max-dn",     type=float, default=1023.0,
                   help="Max DN value for --normalize dn (LISS-IV is 10-bit: 1023)")
    p.add_argument("--chip-size",  type=int, default=512)
    p.add_argument("--stride",     type=int, default=512,
                   help="Stride between chips. Use stride < chip-size for overlap.")
    p.add_argument("--max-nodata", type=float, default=0.30,
                   help="Skip chips where > this fraction of pixels are zero (nodata/border)")
    p.add_argument("--min-cloud",  type=float, default=0.02)
    p.add_argument("--max-cloud",  type=float, default=0.95)
    args = p.parse_args()

    bands = [int(x) for x in args.bands.split(",")] if args.bands else None

    summary = chip_scenes(
        cloudy_dir=Path(args.cloudy_dir),
        clear_dir=Path(args.clear_dir),
        output=Path(args.output),
        mask_dir=Path(args.mask_dir) if args.mask_dir else None,
        bands=bands,
        normalize=args.normalize,
        max_dn=args.max_dn,
        chip_size=args.chip_size,
        stride=args.stride,
        max_nodata=args.max_nodata,
        min_cloud=args.min_cloud,
        max_cloud=args.max_cloud,
    )

    print("\n=== LISS-IV CHIPPING SUMMARY ===")
    print(f"  Scenes processed : {summary['scenes_processed']}")
    print(f"  Chips written    : {summary['chips_written']}")
    print(f"  Skipped (nodata) : {summary['skipped_nodata']}")
    print(f"  Skipped (cloud %%): {summary['skipped_cloud_fraction']}")
    print(f"  Output           : {summary['output']}")
    print("\n  Next steps:")
    print(f"    python -m ai.dataset_tools.audit --dataset {summary['output']}")
    print(f"    python -m ai.restormer.train --config ai/restormer/train_config.yaml "
          f"--resume ai/restormer/checkpoints/last.pt")


if __name__ == "__main__":
    main()
