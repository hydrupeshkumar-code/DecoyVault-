"""
LISS-IV real imagery dataset adapter.

Used for Phase 3 fine-tuning and evaluation on real Resourcesat-2/2A scenes.

On-disk layout:
    root/
        cloudy/   <scene_id>.tif   — [H, W, 3] Green/Red/NIR GeoTIFF
        clear/    <scene_id>.tif   — same scene, cloud-free reference
        masks/    <scene_id>.tif   — cloud mask
        meta/     <scene_id>.json  — acquisition metadata (optional)

Band convention:
    Band 0 → LISS-IV Green (~520–590 nm)
    Band 1 → LISS-IV Red   (~620–680 nm)
    Band 2 → LISS-IV NIR   (~770–860 nm)

LISS-IV specific notes:
    - Spatial resolution: 5.8 m (vs 10 m for Sentinel-2)
    - Radiometric resolution: 10-bit (0–1023 DN)
    - No standard scaling constant — normalize per scene using percentile stretch
    - GeoTIFF metadata (CRS, transform) must be preserved for output
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ai.datasets.base import CloudRemovalDataset, CloudSample, CloudSceneMeta
from ai.geospatial.tiff_io import TiffMeta, load_any, read_tiff


class LISS4Dataset(CloudRemovalDataset):
    """
    Real LISS-IV imagery dataset adapter.

    Args:
        root_dir:     Root of the LISS-IV dataset.
        preserve_meta: Store TiffMeta per sample for GeoTIFF output.
        **kwargs:     Passed to CloudRemovalDataset.__init__.
    """

    def __init__(
        self,
        root_dir: str | Path,
        preserve_meta: bool = True,
        **kwargs,
    ) -> None:
        self._preserve_meta = preserve_meta
        self._tiff_metas: dict[str, TiffMeta] = {}
        super().__init__(root_dir, **kwargs)

    @property
    def dataset_name(self) -> str:
        return "LISS-IV"

    def _collect_samples(self) -> list[str]:
        cloudy_dir = self.root / "cloudy"
        clear_dir  = self.root / "clear"
        mask_dir   = self.root / "masks"

        if not cloudy_dir.exists():
            raise RuntimeError(
                f"[LISS-IV] cloudy/ directory not found at {self.root}. "
                "Place LISS-IV GeoTIFF scenes in cloudy/, clear/, masks/."
            )

        exts = {".tif", ".tiff"}
        cloudy = {p.stem for p in cloudy_dir.iterdir() if p.suffix.lower() in exts}
        clear  = {p.stem for p in clear_dir.iterdir()  if p.suffix.lower() in exts} if clear_dir.exists() else cloudy
        masks  = {p.stem for p in mask_dir.iterdir()   if p.suffix.lower() in exts} if mask_dir.exists() else set()

        # For inference-only (no clear reference), accept cloudy-only
        if masks:
            common = sorted(cloudy & masks)
        else:
            common = sorted(cloudy)

        return common

    def _load_sample(self, stem: str) -> CloudSample:
        cloudy_path = self._find_file(self.root / "cloudy", stem)
        cloudy_arr, tiff_meta = read_tiff(str(cloudy_path), normalize=True)

        if self._preserve_meta and tiff_meta is not None:
            self._tiff_metas[stem] = tiff_meta

        # Clear reference (may not exist for inference-only scenes)
        clear_dir = self.root / "clear"
        if clear_dir.exists():
            try:
                clear_path = self._find_file(clear_dir, stem)
                clear_arr, _ = read_tiff(str(clear_path), normalize=True)
            except FileNotFoundError:
                clear_arr = cloudy_arr.copy()
        else:
            clear_arr = cloudy_arr.copy()

        # Cloud mask
        mask_dir = self.root / "masks"
        if mask_dir.exists():
            try:
                mask_path = self._find_file(mask_dir, stem)
                mask_arr, _ = load_any(mask_path, normalize=False)
                if mask_arr.ndim == 3 and mask_arr.shape[0] > 1:
                    mask_arr = mask_arr[0:1]
                elif mask_arr.ndim == 2:
                    mask_arr = mask_arr[np.newaxis]
                mask_arr = (mask_arr > 0.5).astype(np.float32)
            except FileNotFoundError:
                mask_arr = np.zeros((1, cloudy_arr.shape[1], cloudy_arr.shape[2]), dtype=np.float32)
        else:
            mask_arr = np.zeros((1, cloudy_arr.shape[1], cloudy_arr.shape[2]), dtype=np.float32)

        # Random spatial crop
        cloudy_arr, clear_arr, mask_arr = self._random_crop_chw(
            cloudy_arr, clear_arr, mask_arr, patch_size=self.patch_size
        )

        # Optional: load acquisition metadata
        acq_date: Optional[str] = None
        meta_dir = self.root / "meta"
        if meta_dir.exists():
            meta_file = meta_dir / f"{stem}.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    scene_meta = json.load(f)
                    acq_date = scene_meta.get("acquisition_date")

        return CloudSample(
            cloudy=torch.from_numpy(cloudy_arr),
            clear=torch.from_numpy(clear_arr),
            mask=torch.from_numpy(mask_arr),
            meta=CloudSceneMeta(
                stem=stem,
                dataset_name=self.dataset_name,
                cloud_fraction=float(mask_arr.mean()),
                cloudy_path=str(cloudy_path),
                acquisition_date=acq_date,
            ),
        )

    def get_tiff_meta(self, stem: str) -> Optional[TiffMeta]:
        """Return the stored TiffMeta for a scene (for GeoTIFF output)."""
        return self._tiff_metas.get(stem)

    @staticmethod
    def _find_file(directory: Path, stem: str) -> Path:
        for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
            p = directory / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"No TIFF with stem '{stem}' in {directory}")
