"""
GeoTIFF I/O with full metadata preservation for LISS-IV satellite imagery.

This module is the ONLY correct way to read/write satellite imagery in this
pipeline. Never use PIL/OpenCV/imageio for scientific imagery — they silently
discard radiometric values, dynamic range, CRS, and geotransform.

Supported formats:
    Read:   .tif, .tiff, .TIF, .TIFF (GeoTIFF and plain TIFF)
    Write:  GeoTIFF (LZW-compressed, preserves CRS + transform)

Band convention (LISS-IV):
    Band 0 → Green (~520–590 nm)
    Band 1 → Red   (~620–680 nm)
    Band 2 → NIR   (~770–860 nm)

Normalization contract:
    Internal tensors: float32 [0, 1], shape [C, H, W]
    On-disk GeoTIFF:  float32 DN or TOA reflectance, shape [H, W, C] (rasterio CHW)

Usage:
    from ai.geospatial.tiff_io import read_tiff, write_tiff, TiffMeta

    arr, meta = read_tiff("scene.tif", band_indices=[2, 3, 7])
    # arr: np.ndarray [3, H, W] float32 in [0,1]
    # meta: TiffMeta (crs, transform, nodata, dtype_orig, band_descriptions)

    write_tiff("output.tif", arr, meta)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

try:
    import rasterio
    from rasterio.transform import Affine
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False
    warnings.warn(
        "rasterio not installed. GeoTIFF spatial metadata (CRS, transform) will "
        "not be preserved. Install it: pip install rasterio",
        ImportWarning,
        stacklevel=2,
    )


@dataclass
class TiffMeta:
    """Container for geospatial metadata attached to a TIFF scene."""

    crs: Optional[object] = None
    transform: Optional[object] = None
    nodata: Optional[float] = None
    dtype_orig: str = "float32"
    width: int = 0
    height: int = 0
    count: int = 3
    band_descriptions: list[str] = field(default_factory=lambda: ["Green", "Red", "NIR"])
    tags: dict = field(default_factory=dict)

    def is_georeferenced(self) -> bool:
        return self.crs is not None and self.transform is not None


def read_tiff(
    path: str | Path,
    band_indices: Sequence[int] | None = None,
    normalize: bool = True,
    norm_percentile: float = 2.0,
) -> tuple[np.ndarray, TiffMeta]:
    """
    Read a GeoTIFF and return a float32 array with metadata.

    Args:
        path:            Path to the .tif file.
        band_indices:    0-indexed band selection from the file.
                         None → read all bands.
        normalize:       If True, normalize to [0,1] using percentile stretch.
        norm_percentile: Lower/upper percentile for stretch (default 2%).

    Returns:
        arr:  [C, H, W] float32 array. Values in [0,1] if normalize=True.
        meta: TiffMeta with all preserved spatial metadata.

    Raises:
        ImportError: if rasterio is not installed.
        FileNotFoundError: if path does not exist.
        ValueError: if band_indices are out of range.
    """
    if not _HAS_RASTERIO:
        raise ImportError(
            "rasterio is required for GeoTIFF I/O. pip install rasterio"
        )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TIFF not found: {path}")

    with rasterio.open(str(path)) as src:
        meta = TiffMeta(
            crs=src.crs,
            transform=src.transform,
            nodata=src.nodata,
            dtype_orig=str(src.dtypes[0]),
            width=src.width,
            height=src.height,
            count=src.count,
            tags=dict(src.tags()),
        )

        # Band descriptions
        descs = src.descriptions
        if descs and any(d for d in descs):
            meta.band_descriptions = [d or f"Band_{i+1}" for i, d in enumerate(descs)]

        if band_indices is not None:
            # rasterio bands are 1-indexed
            rasterio_bands = [i + 1 for i in band_indices]
            if any(b > src.count for b in rasterio_bands):
                raise ValueError(
                    f"band_indices {band_indices} out of range for file with "
                    f"{src.count} bands."
                )
            data = src.read(rasterio_bands).astype(np.float32)  # [C, H, W]
            meta.band_descriptions = [
                meta.band_descriptions[i] if i < len(meta.band_descriptions) else f"Band_{i}"
                for i in band_indices
            ]
            meta.count = len(band_indices)
        else:
            data = src.read().astype(np.float32)  # [C, H, W]

    # Replace nodata with NaN before normalization
    if meta.nodata is not None:
        data[data == meta.nodata] = np.nan

    if normalize:
        data = _percentile_normalize(data, norm_percentile)

    # CRITICAL: nodata pixels were set to NaN above (and percentile-normalize
    # ignores NaN when computing the stretch, but leaves NaN in the output).
    # NaN must never reach the model — it produces NaN losses/gradients in
    # training and NaN outputs in inference. Replace any remaining NaN with 0.
    data = np.nan_to_num(data, nan=0.0).astype(np.float32)

    return data, meta


def write_tiff(
    path: str | Path,
    data: np.ndarray,
    meta: TiffMeta,
    compress: str = "lzw",
) -> None:
    """
    Write a float32 array to a GeoTIFF, preserving CRS and geotransform.

    Args:
        path:     Output file path (.tif).
        data:     [C, H, W] float32 array.
        meta:     TiffMeta from the source scene.
        compress: Compression algorithm ('lzw', 'deflate', or 'none').

    Raises:
        ImportError: if rasterio is not installed.
    """
    if not _HAS_RASTERIO:
        raise ImportError("rasterio is required for GeoTIFF output. pip install rasterio")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = data[np.newaxis]

    C, H, W = data.shape

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": C,
        "height": H,
        "width": W,
        "compress": compress if compress != "none" else None,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    if meta.crs is not None:
        profile["crs"] = meta.crs
    if meta.transform is not None:
        profile["transform"] = meta.transform

    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(data)
        for i, desc in enumerate(meta.band_descriptions[:C]):
            dst.update_tags(i + 1, description=desc)
        if meta.tags:
            dst.update_tags(**meta.tags)


def write_tiff_from_array(
    path: str | Path,
    data: np.ndarray,
    band_descriptions: list[str] | None = None,
) -> None:
    """
    Write a plain (non-georeferenced) float32 GeoTIFF.

    Useful for saving predictions when no source metadata is available.
    """
    meta = TiffMeta(
        width=data.shape[-1] if data.ndim == 3 else data.shape[1],
        height=data.shape[-2] if data.ndim == 3 else data.shape[0],
        count=data.shape[0] if data.ndim == 3 else 1,
        band_descriptions=band_descriptions or ["Green", "Red", "NIR"],
    )
    write_tiff(path, data, meta)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _percentile_normalize(
    data: np.ndarray,
    percentile: float = 2.0,
) -> np.ndarray:
    """
    Per-band percentile stretch to [0, 1], ignoring NaN.

    This is the scientifically correct normalization for satellite imagery.
    A fixed division by 10000 (Sentinel-2 DN → reflectance) is NOT used here
    because LISS-IV DN values and scaling differ from Sentinel-2.
    """
    result = np.empty_like(data)
    for c in range(data.shape[0]):
        band = data[c]
        valid = band[~np.isnan(band)]
        if len(valid) == 0:
            result[c] = 0.0
            continue
        lo = np.percentile(valid, percentile)
        hi = np.percentile(valid, 100.0 - percentile)
        if hi == lo:
            result[c] = 0.0
        else:
            stretched = (band - lo) / (hi - lo)
            result[c] = np.clip(stretched, 0.0, 1.0)
    return result.astype(np.float32)


def load_any(
    path: str | Path,
    band_indices: Sequence[int] | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, Optional[TiffMeta]]:
    """
    Unified loader: handles .npy (legacy) and .tif/.tiff (preferred).

    Returns (array [C, H, W] float32, meta or None).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        return read_tiff(path, band_indices=band_indices, normalize=normalize)

    if suffix == ".npy":
        arr = np.load(str(path)).astype(np.float32)
        # Legacy: may be [H, W, C] — normalize to [C, H, W]
        if arr.ndim == 3 and arr.shape[2] <= 13:
            arr = arr.transpose(2, 0, 1)
        if band_indices is not None:
            arr = arr[band_indices]
        if normalize:
            arr = np.clip(arr, 0.0, 1.0)
        return arr, None

    raise ValueError(f"Unsupported file format: {suffix}. Use .tif or .npy")
