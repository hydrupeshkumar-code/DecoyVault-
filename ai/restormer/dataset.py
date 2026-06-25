"""
SEN2-CR dataset loader for Restormer cloud removal training.

Expects the following on-disk layout (configurable via root_dir):

    root_dir/
        cloudy/   *.npy or *.tif  – shape [H, W, 13] or [H, W, 3]
        clear/    *.npy or *.tif  – shape [H, W, 13] or [H, W, 3]
        masks/    *.npy or *.tif  – shape [H, W]      binary cloud mask

Band selection for LISS-IV simulation:
    SEN2-CR B3 → Green  (index 2)
    SEN2-CR B4 → Red    (index 3)
    SEN2-CR B8 → NIR    (index 7)

All values are expected in [0, 1] after normalisation.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False


# Band indices inside a 13-band SEN2-CR file that map to Green/Red/NIR
SEN2_BAND_INDICES: list[int] = [2, 3, 7]


def _load_array(path: Path) -> np.ndarray:
    """Load a numpy .npy or GeoTIFF file as a float32 array."""
    if path.suffix == ".npy":
        return np.load(str(path)).astype(np.float32)
    if path.suffix in (".tif", ".tiff"):
        if not _HAS_RASTERIO:
            raise ImportError("rasterio is required to load .tif files. pip install rasterio")
        with rasterio.open(str(path)) as src:
            data = src.read().astype(np.float32)  # [C, H, W]
            return data.transpose(1, 2, 0)         # → [H, W, C]
    raise ValueError(f"Unsupported file format: {path.suffix}")


class SEN2CRDataset(Dataset):
    """
    SEN2-CR cloud removal dataset.

    Args:
        root_dir:     Root of the dataset split (e.g. 'datasets/sen2cr').
        split:        'train', 'val', or 'test'. Used to locate sub-folders
                      if split_dirs is provided; otherwise all samples are used.
        patch_size:   Spatial crop size for training patches.
        augment:      Apply random horizontal/vertical flips and rotation.
        band_indices: Which bands to extract from the multi-band files.
        transform:    Optional callable applied jointly to (cloudy, clear, mask).
        max_cloud_cover: Skip samples where mean(mask) < threshold (0–1).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        patch_size: int = 256,
        augment: bool = True,
        band_indices: list[int] | None = None,
        transform: Optional[Callable] = None,
        max_cloud_cover: float = 0.0,
    ) -> None:
        super().__init__()
        self.root = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and (split == "train")
        self.band_indices = band_indices if band_indices is not None else SEN2_BAND_INDICES
        self.transform = transform
        self.max_cloud_cover = max_cloud_cover

        self.cloudy_dir = self.root / "cloudy"
        self.clear_dir = self.root / "clear"
        self.mask_dir = self.root / "masks"

        self.samples: list[str] = self._collect_samples()

    # ------------------------------------------------------------------

    def _collect_samples(self) -> list[str]:
        """Return sorted list of base names present in all three folders."""
        cloudy_names = {p.stem for p in self.cloudy_dir.iterdir() if not p.name.startswith(".")}
        clear_names = {p.stem for p in self.clear_dir.iterdir() if not p.name.startswith(".")}
        mask_names = {p.stem for p in self.mask_dir.iterdir() if not p.name.startswith(".")}
        common = sorted(cloudy_names & clear_names & mask_names)
        if not common:
            raise RuntimeError(
                f"No matching samples found under {self.root}. "
                "Ensure cloudy/, clear/, and masks/ directories share file stems."
            )
        return common

    def _find_file(self, directory: Path, stem: str) -> Path:
        for ext in (".npy", ".tif", ".tiff"):
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No file with stem '{stem}' in {directory}")

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        stem = self.samples[idx]

        cloudy_raw = _load_array(self._find_file(self.cloudy_dir, stem))
        clear_raw = _load_array(self._find_file(self.clear_dir, stem))
        mask_raw = _load_array(self._find_file(self.mask_dir, stem))

        # Select bands if multi-band
        cloudy = self._select_bands(cloudy_raw)   # [H, W, 3]
        clear = self._select_bands(clear_raw)      # [H, W, 3]

        # Mask: ensure [H, W]
        if mask_raw.ndim == 3:
            mask_raw = mask_raw[..., 0]
        mask = (mask_raw > 0.5).astype(np.float32)  # binary

        # Skip very low cloud cover tiles during training
        if self.augment and mask.mean() < self.max_cloud_cover:
            # Resample another index
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)

        # Random crop
        cloudy, clear, mask = self._random_crop(cloudy, clear, mask)

        # Augmentation
        if self.augment:
            cloudy, clear, mask = self._augment(cloudy, clear, mask)

        # Normalise to [0, 1] per channel (clip any stray values)
        cloudy = np.clip(cloudy, 0.0, 1.0)
        clear = np.clip(clear, 0.0, 1.0)

        # → torch tensors [C, H, W]
        cloudy_t = torch.from_numpy(cloudy.transpose(2, 0, 1))
        clear_t = torch.from_numpy(clear.transpose(2, 0, 1))
        mask_t = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

        if self.transform is not None:
            cloudy_t, clear_t, mask_t = self.transform(cloudy_t, clear_t, mask_t)

        return {
            "cloudy": cloudy_t,
            "clear": clear_t,
            "mask": mask_t,
            "stem": stem,
        }

    # ------------------------------------------------------------------

    def _select_bands(self, arr: np.ndarray) -> np.ndarray:
        """Select Green/Red/NIR bands from a multi-band array."""
        if arr.ndim == 2:
            # Single-band: replicate
            return np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 3:
            return arr
        # Multi-band → select
        return arr[..., self.band_indices]

    def _random_crop(
        self,
        cloudy: np.ndarray,
        clear: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        H, W = cloudy.shape[:2]
        p = self.patch_size
        if H <= p or W <= p:
            # Pad if image is smaller than patch
            pad_h = max(0, p - H + 1)
            pad_w = max(0, p - W + 1)
            cloudy = np.pad(cloudy, ((0, pad_h), (0, pad_w), (0, 0)))
            clear = np.pad(clear, ((0, pad_h), (0, pad_w), (0, 0)))
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)))
            H, W = cloudy.shape[:2]
        top = random.randint(0, H - p)
        left = random.randint(0, W - p)
        return (
            cloudy[top : top + p, left : left + p],
            clear[top : top + p, left : left + p],
            mask[top : top + p, left : left + p],
        )

    def _augment(
        self,
        cloudy: np.ndarray,
        clear: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Random horizontal flip
        if random.random() > 0.5:
            cloudy = np.fliplr(cloudy).copy()
            clear = np.fliplr(clear).copy()
            mask = np.fliplr(mask).copy()
        # Random vertical flip
        if random.random() > 0.5:
            cloudy = np.flipud(cloudy).copy()
            clear = np.flipud(clear).copy()
            mask = np.flipud(mask).copy()
        # Random 90° rotation
        k = random.randint(0, 3)
        if k > 0:
            cloudy = np.rot90(cloudy, k).copy()
            clear = np.rot90(clear, k).copy()
            mask = np.rot90(mask, k).copy()
        return cloudy, clear, mask
