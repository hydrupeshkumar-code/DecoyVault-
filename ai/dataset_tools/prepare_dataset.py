"""
Dataset preparation + cloud-mask generation for cloud-removal training.

WHY THIS EXISTS
---------------
The raw public datasets do not drop straight into the training pipeline:

  * SEN12MS-CR ships cloudy + cloud-free Sentinel-2 tiles but **no cloud masks**,
    and the flat-layout loader (ai/datasets/sen12ms_cr.py) *requires* a masks/
    directory whose stems match cloudy/ and clear/. Without masks, zero samples
    are collected and training never starts.

  * The native SEN12MS-CR / RICE folder structures use non-matching filenames
    for the cloudy and clear members of a pair, so the loaders can't pair them.

This tool normalises ANY (cloudy, clear[, mask]) source into the exact flat
layout both loaders expect:

    <output>/
        cloudy/  pair_000001.npy   [H, W, 3] float32 in [0,1]  (Green, Red, NIR)
        clear/   pair_000001.npy   [H, W, 3] float32 in [0,1]
        masks/   pair_000001.npy   [H, W]    float32 binary    (1 = cloud)

The mask, when not supplied, is derived from the cloudy-vs-clear brightness
difference (clouds are bright and raise reflectance over the clear reference).
This is sufficient because the model never consumes the mask during training
(see ai/restormer/train.py — the loss is criterion(pred, clear)); masks only
drive cloud-fraction filtering and cloud-region metric reporting.

EXAMPLES
--------
SEN12MS-CR (13-band Sentinel-2, slice B3/B4/B8 -> Green/Red/NIR):
    python -m ai.dataset_tools.prepare_dataset \
        --cloudy-dir  raw/SEN12MSCR/s2_cloudy \
        --clear-dir   raw/SEN12MSCR/s2_cloudfree \
        --output      datasets/sen12ms_cr \
        --bands 2,3,7 --normalize scale --scale 10000 \
        --match order --max 4000

RICE2 (RGB pngs that already ship masks):
    python -m ai.dataset_tools.prepare_dataset \
        --cloudy-dir raw/RICE2/cloud \
        --clear-dir  raw/RICE2/label \
        --mask-dir   raw/RICE2/mask \
        --output     datasets/rice2 \
        --normalize auto --match order

After running, audit then train:
    python -m ai.dataset_tools.audit  --dataset <output>
    python -m ai.restormer.train --smoke --config ai/restormer/train_config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

_RASTER_EXTS = (".tif", ".tiff", ".npy")
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
_ALL_EXTS = _RASTER_EXTS + _IMAGE_EXTS


# ---------------------------------------------------------------------------
# Reading (handles GeoTIFF, .npy, and ordinary 8-bit images for smoke sets)
# ---------------------------------------------------------------------------

def _read_raster(path: Path) -> np.ndarray:
    """Read any supported file as a [C, H, W] float32 array (raw, un-normalised)."""
    suffix = path.suffix.lower()
    if suffix in _RASTER_EXTS:
        from ai.geospatial.tiff_io import load_any
        arr, _ = load_any(path, normalize=False)  # [C, H, W]
        return arr.astype(np.float32)
    if suffix in _IMAGE_EXTS:
        # PIL is acceptable here: these are smoke-test inputs, not science data.
        from PIL import Image
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)  # [H, W, 3]
        return img.transpose(2, 0, 1)  # [C, H, W]
    raise ValueError(f"Unsupported file format: {suffix}")


def _discover(root: Path) -> list[Path]:
    """Recursively find all readable rasters/images under root, sorted."""
    if root.is_file():
        return [root]
    files = [p for p in root.rglob("*") if p.suffix.lower() in _ALL_EXTS and p.is_file()]
    return sorted(files)


# ---------------------------------------------------------------------------
# Normalisation + band selection
# ---------------------------------------------------------------------------

def _select_bands(arr: np.ndarray, bands: Optional[list[int]]) -> np.ndarray:
    """Select 3 bands (Green, Red, NIR) from a [C, H, W] array."""
    C = arr.shape[0]
    if C == 3:
        return arr
    if bands is not None:
        if C < max(bands) + 1:
            raise ValueError(f"Cannot select bands {bands} from a {C}-band array.")
        return arr[bands]
    if C == 1:
        return np.repeat(arr, 3, axis=0)
    if C >= 3:
        log.warning("No --bands given for a %d-band file; keeping the first 3.", C)
        return arr[:3]
    raise ValueError(f"Cannot produce 3 bands from a {C}-band array.")


def _normalize(arr: np.ndarray, mode: str, scale: float) -> np.ndarray:
    """Normalise a [C, H, W] array to float32 in [0, 1]."""
    if mode == "auto":
        mx = float(np.nanmax(arr)) if arr.size else 0.0
        if mx <= 1.5:
            mode = "none"
        elif mx <= 255.0:
            mode = "scale255"
        else:
            mode = "scale"

    if mode == "none":
        out = arr
    elif mode == "scale":
        out = arr / scale
    elif mode == "scale255":
        out = arr / 255.0
    elif mode == "percentile":
        from ai.geospatial.tiff_io import _percentile_normalize
        out = _percentile_normalize(arr)
    elif mode == "minmax":
        lo = np.nanmin(arr)
        hi = np.nanmax(arr)
        out = (arr - lo) / (hi - lo + 1e-8)
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")

    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Cloud-mask derivation
# ---------------------------------------------------------------------------

def _derive_mask(
    cloudy: np.ndarray,
    clear: np.ndarray,
    min_delta: float,
    bright_abs: float,
) -> np.ndarray:
    """
    Derive a binary cloud mask [H, W] from a normalised (cloudy, clear) pair.

    Clouds raise reflectance, so a cloud pixel is either:
      (a) notably brighter in the cloudy image than in the clear reference, or
      (b) very bright in absolute terms (thick/opaque cloud).

    Both inputs are [3, H, W] float32 in [0, 1].
    """
    cloudy_gray = cloudy.mean(axis=0)
    clear_gray = clear.mean(axis=0)
    diff = cloudy_gray - clear_gray

    mask = ((diff > min_delta) | (cloudy_gray > bright_abs)).astype(np.float32)

    # Morphological cleanup: close small gaps (closing), then drop isolated
    # speckle (opening). scipy.ndimage has a stable API across versions.
    try:
        from scipy import ndimage as ndi
        struct = np.ones((3, 3), dtype=bool)
        m = ndi.binary_closing(mask > 0.5, structure=struct)
        m = ndi.binary_opening(m, structure=struct)
        mask = m.astype(np.float32)
    except Exception:  # pragma: no cover - cleanup is best-effort
        pass

    return mask


def _load_mask_file(path: Path, hw: tuple[int, int]) -> np.ndarray:
    """Load a supplied mask file and binarise it to [H, W] float32."""
    arr = _read_raster(path)
    m = arr[0] if arr.ndim == 3 else arr
    m = (m > (0.5 * float(m.max()) if m.max() > 1.5 else 0.5)).astype(np.float32)
    if m.shape != hw:
        log.warning("Mask %s shape %s != image %s; using image-derived fallback.", path.name, m.shape, hw)
        return np.zeros(hw, dtype=np.float32)
    return m


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

def _pair(
    cloudy_files: list[Path],
    clear_files: list[Path],
    mask_files: Optional[list[Path]],
    match: str,
) -> list[tuple[Path, Path, Optional[Path]]]:
    """Pair cloudy/clear[/mask] files by 'order' (zip) or 'stem' (intersection)."""
    if match == "order":
        n = min(len(cloudy_files), len(clear_files))
        if len(cloudy_files) != len(clear_files):
            log.warning(
                "order-match: %d cloudy vs %d clear files; pairing the first %d.",
                len(cloudy_files), len(clear_files), n,
            )
        masks = mask_files[:n] if mask_files else [None] * n
        if mask_files and len(mask_files) < n:
            masks = mask_files + [None] * (n - len(mask_files))
        return list(zip(cloudy_files[:n], clear_files[:n], masks))

    if match == "stem":
        clear_by_stem = {p.stem: p for p in clear_files}
        mask_by_stem = {p.stem: p for p in (mask_files or [])}
        pairs: list[tuple[Path, Path, Optional[Path]]] = []
        for c in cloudy_files:
            if c.stem in clear_by_stem:
                pairs.append((c, clear_by_stem[c.stem], mask_by_stem.get(c.stem)))
        if not pairs:
            raise RuntimeError(
                "stem-match found no common stems between cloudy/ and clear/. "
                "Use --match order if the cloudy and clear filenames differ."
            )
        return pairs

    raise ValueError(f"Unknown --match mode: {match}")


# ---------------------------------------------------------------------------
# Main preparation routine
# ---------------------------------------------------------------------------

def prepare(
    cloudy_dir: Path,
    clear_dir: Path,
    output: Path,
    mask_dir: Optional[Path] = None,
    bands: Optional[list[int]] = None,
    normalize: str = "auto",
    scale: float = 10_000.0,
    match: str = "order",
    min_delta: float = 0.04,
    bright_abs: float = 0.55,
    max_pairs: Optional[int] = None,
) -> dict:
    """Build a flat cloud-removal dataset and return a summary dict."""
    cloudy_files = _discover(cloudy_dir)
    clear_files = _discover(clear_dir)
    mask_files = _discover(mask_dir) if mask_dir else None

    if not cloudy_files:
        raise RuntimeError(f"No readable files found under --cloudy-dir {cloudy_dir}")
    if not clear_files:
        raise RuntimeError(f"No readable files found under --clear-dir {clear_dir}")

    pairs = _pair(cloudy_files, clear_files, mask_files, match)
    if max_pairs:
        pairs = pairs[:max_pairs]

    out_cloudy = output / "cloudy"
    out_clear = output / "clear"
    out_masks = output / "masks"
    for d in (out_cloudy, out_clear, out_masks):
        d.mkdir(parents=True, exist_ok=True)

    cloud_fractions: list[float] = []
    written = 0
    skipped = 0

    log.info("Preparing %d candidate pairs -> %s", len(pairs), output)
    for i, (c_path, cl_path, m_path) in enumerate(pairs):
        stem = f"pair_{i + 1:06d}"
        try:
            cloudy = _normalize(_select_bands(_read_raster(c_path), bands), normalize, scale)
            clear = _normalize(_select_bands(_read_raster(cl_path), bands), normalize, scale)
        except Exception as e:
            log.warning("Skipping pair %d (%s / %s): %s", i + 1, c_path.name, cl_path.name, e)
            skipped += 1
            continue

        if cloudy.shape != clear.shape:
            log.warning("Skipping pair %d: shape mismatch %s vs %s", i + 1, cloudy.shape, clear.shape)
            skipped += 1
            continue

        hw = cloudy.shape[1:]
        if m_path is not None:
            mask = _load_mask_file(m_path, hw)
        else:
            mask = _derive_mask(cloudy, clear, min_delta, bright_abs)

        cloud_fractions.append(float(mask.mean()))

        # Write [H, W, C] for images, [H, W] for masks (consumed by both loaders).
        np.save(out_cloudy / f"{stem}.npy", cloudy.transpose(1, 2, 0))
        np.save(out_clear / f"{stem}.npy", clear.transpose(1, 2, 0))
        np.save(out_masks / f"{stem}.npy", mask)
        written += 1

        if (i + 1) % 200 == 0:
            log.info("  ... %d/%d written", written, len(pairs))

    cf = np.array(cloud_fractions) if cloud_fractions else np.zeros(1)
    summary = {
        "output": str(output),
        "pairs_written": written,
        "pairs_skipped": skipped,
        "cloud_fraction_mean": float(cf.mean()),
        "cloud_fraction_p10": float(np.percentile(cf, 10)),
        "cloud_fraction_p90": float(np.percentile(cf, 90)),
        "in_useful_range_0.02_0.95": int(np.sum((cf >= 0.02) & (cf <= 0.95))),
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_bands(value: Optional[str]) -> Optional[list[int]]:
    if not value:
        return None
    return [int(x) for x in value.split(",") if x.strip() != ""]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Prepare a flat cloud-removal dataset + masks")
    p.add_argument("--cloudy-dir", required=True, help="Directory (searched recursively) of cloudy rasters")
    p.add_argument("--clear-dir", required=True, help="Directory (searched recursively) of clear rasters")
    p.add_argument("--mask-dir", default=None, help="Optional directory of existing masks")
    p.add_argument("--output", default="datasets/sen12ms_cr", help="Output dataset root")
    p.add_argument("--bands", default=None,
                   help="Comma-separated 0-indexed bands -> Green,Red,NIR (e.g. '2,3,7' for Sentinel-2)")
    p.add_argument("--normalize", default="auto",
                   choices=["auto", "scale", "scale255", "percentile", "minmax", "none"],
                   help="How to bring raw values into [0,1]")
    p.add_argument("--scale", type=float, default=10_000.0, help="Divisor for --normalize scale (S2 L1C = 10000)")
    p.add_argument("--match", default="order", choices=["order", "stem"],
                   help="Pair cloudy<->clear by sort order (default) or by matching filename stem")
    p.add_argument("--min-delta", type=float, default=0.04, help="Cloudy-minus-clear brightness delta for masks")
    p.add_argument("--bright-abs", type=float, default=0.55, help="Absolute brightness above which a pixel is cloud")
    p.add_argument("--max", type=int, default=None, help="Limit number of pairs (use for subsets)")
    args = p.parse_args()

    summary = prepare(
        cloudy_dir=Path(args.cloudy_dir),
        clear_dir=Path(args.clear_dir),
        output=Path(args.output),
        mask_dir=Path(args.mask_dir) if args.mask_dir else None,
        bands=_parse_bands(args.bands),
        normalize=args.normalize,
        scale=args.scale,
        match=args.match,
        min_delta=args.min_delta,
        bright_abs=args.bright_abs,
        max_pairs=args.max,
    )

    print("\n=== DATASET PREPARATION SUMMARY ===")
    print(f"  Output:              {summary['output']}")
    print(f"  Pairs written:       {summary['pairs_written']}")
    print(f"  Pairs skipped:       {summary['pairs_skipped']}")
    print(f"  Cloud fraction:      {summary['cloud_fraction_mean']:.3f} "
          f"(p10={summary['cloud_fraction_p10']:.3f}, p90={summary['cloud_fraction_p90']:.3f})")
    print(f"  In useful range:     {summary['in_useful_range_0.02_0.95']}/{summary['pairs_written']}")
    print("\n  Next:")
    print(f"    python -m ai.dataset_tools.audit --dataset {summary['output']}")
    print(f"    python -m ai.restormer.train --smoke --config ai/restormer/train_config.yaml")


if __name__ == "__main__":
    main()
