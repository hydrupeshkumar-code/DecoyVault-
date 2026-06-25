"""
SEN12MS-CR dataset adapter — PRIMARY training dataset.

SEN12MS-CR provides globally distributed Sentinel-1 SAR + Sentinel-2 MSI
cloudy/clear patch pairs, with cloud masks derived from s2cloudless.

On-disk layout:
    root/
        ROIs<id>_<season>/
            S2_cloudy/  <tile_id>.tif   — 13-band Sentinel-2
            S2_clear/   <tile_id>.tif   — 13-band Sentinel-2 (cloud-free)
            S1/         <tile_id>.tif   — 2-band SAR (optional, not used here)
            LC/         <tile_id>.tif   — land cover labels (optional)
            DEM/        <tile_id>.tif   — elevation (optional)
            Mask/       <tile_id>.tif   — cloud mask

Or the flat SEN12MS-CR-TS layout:
    root/
        cloudy/<tile_id>.tif
        clear/<tile_id>.tif
        masks/<tile_id>.tif

LISS-IV simulation: S2 bands [B3, B4, B8] → Green, Red, NIR.
S2 DN → TOA reflectance: divide by 10000 (standard Sentinel-2 L1C scaling).

Reference:
    Ebel et al. (2022) "SEN12MS-CR-TS: A Remote-Sensing Data Set for Multi-Modal
    Multi-Temporal Cloud Removal", IEEE TGRS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ai.datasets.base import CloudRemovalDataset, CloudSample, CloudSceneMeta
from ai.geospatial.tiff_io import load_any

# S2 band indices for LISS-IV simulation (0-indexed within 13-band file)
S2_GREEN = 2   # B3 ~560 nm
S2_RED   = 3   # B4 ~665 nm
S2_NIR   = 7   # B8 ~842 nm
S2_LISS4_BANDS = [S2_GREEN, S2_RED, S2_NIR]

# Standard Sentinel-2 L1C DN scaling factor
S2_SCALE = 10_000.0


class SEN12MSCRDataset(CloudRemovalDataset):
    """
    SEN12MS-CR cloud removal dataset adapter.

    Supports both the hierarchical ROI layout and the flat layout.

    Args:
        root_dir:   Root of the SEN12MS-CR dataset (or a pre-processed flat layout).
        layout:     'flat' (cloudy/clear/masks/) or 'roi' (ROIs<id>_<season>/).
        band_indices: Which S2 bands to select. Default = [B3, B4, B8] = LISS-IV.
        s2_scale:   DN scaling divisor (10000 for L1C reflectance).
        **kwargs:   Passed to CloudRemovalDataset.__init__.
    """

    def __init__(
        self,
        root_dir: str | Path,
        layout: str = "flat",
        band_indices: list[int] | None = None,
        s2_scale: float = S2_SCALE,
        **kwargs,
    ) -> None:
        self._layout = layout
        self._band_indices = band_indices if band_indices is not None else S2_LISS4_BANDS
        self._s2_scale = s2_scale
        super().__init__(root_dir, **kwargs)

    @property
    def dataset_name(self) -> str:
        return "SEN12MS-CR"

    # ------------------------------------------------------------------
    # Sample discovery
    # ------------------------------------------------------------------

    def _collect_samples(self) -> list[str]:
        if self._layout == "flat":
            return self._collect_flat()
        return self._collect_roi()

    def _collect_flat(self) -> list[str]:
        """Flat layout: cloudy/ clear/ masks/ all share the same stems."""
        cloudy_dir = self.root / "cloudy"
        clear_dir  = self.root / "clear"
        mask_dir   = self.root / "masks"

        for d in (cloudy_dir, clear_dir, mask_dir):
            if not d.exists():
                raise RuntimeError(
                    f"[SEN12MS-CR] Expected directory not found: {d}. "
                    "Check that root_dir points to the flat-layout dataset."
                )

        cloudy = {p.stem for p in cloudy_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff", ".npy")}
        clear  = {p.stem for p in clear_dir.iterdir()  if p.suffix.lower() in (".tif", ".tiff", ".npy")}
        masks  = {p.stem for p in mask_dir.iterdir()   if p.suffix.lower() in (".tif", ".tiff", ".npy")}
        common = sorted(cloudy & clear & masks)

        if not common:
            raise RuntimeError(
                f"[SEN12MS-CR] No common stems found across cloudy/clear/masks in {self.root}."
            )
        return common

    def _collect_roi(self) -> list[str]:
        """ROI layout: ROIs<id>_<season>/S2_cloudy|S2_clear|Mask."""
        stems: list[str] = []
        for roi_dir in sorted(self.root.iterdir()):
            if not roi_dir.is_dir() or not roi_dir.name.startswith("ROIs"):
                continue
            cloudy_dir = roi_dir / "S2_cloudy"
            clear_dir  = roi_dir / "S2_clear"
            mask_dir   = roi_dir / "Mask"
            if not all(d.exists() for d in (cloudy_dir, clear_dir, mask_dir)):
                continue
            c = {p.stem for p in cloudy_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")}
            cl = {p.stem for p in clear_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")}
            m  = {p.stem for p in mask_dir.iterdir()  if p.suffix.lower() in (".tif", ".tiff")}
            for stem in sorted(c & cl & m):
                # Encode ROI dir into stem to avoid collisions
                stems.append(f"{roi_dir.name}/{stem}")
        return stems

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def _load_sample(self, stem: str) -> CloudSample:
        if self._layout == "flat":
            return self._load_flat(stem)
        return self._load_roi(stem)

    def _load_flat(self, stem: str) -> CloudSample:
        cloudy_arr = self._load_bands(self._find_file(self.root / "cloudy", stem))
        clear_arr  = self._load_bands(self._find_file(self.root / "clear", stem))
        mask_arr   = self._load_mask(self._find_file(self.root / "masks", stem))

        cloudy_arr, clear_arr, mask_arr = self._random_crop_chw(
            cloudy_arr, clear_arr, mask_arr, patch_size=self.patch_size
        )

        cloud_fraction = float(mask_arr.mean())

        return CloudSample(
            cloudy=torch.from_numpy(cloudy_arr),
            clear=torch.from_numpy(clear_arr),
            mask=torch.from_numpy(mask_arr),
            meta=CloudSceneMeta(
                stem=stem,
                dataset_name=self.dataset_name,
                cloud_fraction=cloud_fraction,
                cloudy_path=str(self.root / "cloudy" / stem),
                clear_path=str(self.root / "clear" / stem),
                mask_path=str(self.root / "masks" / stem),
            ),
        )

    def _load_roi(self, stem: str) -> CloudSample:
        roi_name, tile_stem = stem.rsplit("/", 1)
        roi_dir = self.root / roi_name
        cloudy_arr = self._load_bands(self._find_file(roi_dir / "S2_cloudy", tile_stem))
        clear_arr  = self._load_bands(self._find_file(roi_dir / "S2_clear", tile_stem))
        mask_arr   = self._load_mask(self._find_file(roi_dir / "Mask", tile_stem))

        cloudy_arr, clear_arr, mask_arr = self._random_crop_chw(
            cloudy_arr, clear_arr, mask_arr, patch_size=self.patch_size
        )

        return CloudSample(
            cloudy=torch.from_numpy(cloudy_arr),
            clear=torch.from_numpy(clear_arr),
            mask=torch.from_numpy(mask_arr),
            meta=CloudSceneMeta(
                stem=stem,
                dataset_name=self.dataset_name,
                cloud_fraction=float(mask_arr.mean()),
            ),
        )

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_bands(self, path: Path) -> np.ndarray:
        """
        Load and select LISS-IV-simulating bands from a multi-band S2 file.

        Returns [3, H, W] float32 in [0, 1].
        """
        arr, _ = load_any(path, normalize=False)  # [C, H, W] float32

        # S2 L1C DN → TOA reflectance
        arr = arr / self._s2_scale

        # Band selection
        if arr.shape[0] >= max(self._band_indices) + 1:
            arr = arr[self._band_indices]
        elif arr.shape[0] == 3:
            pass  # already 3-band
        else:
            raise ValueError(
                f"Cannot select bands {self._band_indices} from {arr.shape[0]}-band file {path}"
            )

        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def _load_mask(self, path: Path) -> np.ndarray:
        """Load binary cloud mask. Returns [1, H, W] float32."""
        arr, _ = load_any(path, normalize=False)
        if arr.ndim == 3:
            arr = arr[0:1]  # take first band [1, H, W]
        else:
            arr = arr[np.newaxis]
        return (arr > 0.5).astype(np.float32)

    @staticmethod
    def _find_file(directory: Path, stem: str) -> Path:
        for ext in (".tif", ".tiff", ".npy"):
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No file with stem '{stem}' in {directory}")
