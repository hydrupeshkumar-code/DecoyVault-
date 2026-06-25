"""
Abstract base class for all cloud-removal dataset adapters.

Every dataset (SEN12MS-CR, SEN2-CR, RICE95, LISS-IV) must implement this
interface so the training pipeline is completely dataset-agnostic.

Design principles:
  - Band ordering is always (Green, Red, NIR) in output tensors.
  - Values are always float32 in [0, 1] at the model boundary.
  - Metadata (source path, acquisition date, cloud fraction) flows through
    as side-channel data for debugging and quality auditing.
  - Augmentation is decoupled from loading and lives in a separate transform.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class CloudSceneMeta:
    """
    Metadata returned alongside every training sample.

    Not used by the model — used for dataset auditing and debugging.
    """
    stem: str = ""
    dataset_name: str = ""
    cloud_fraction: float = 0.0
    cloudy_path: str = ""
    clear_path: str = ""
    mask_path: str = ""
    pair_quality_score: float = 1.0
    acquisition_date: Optional[str] = None
    extra: dict = field(default_factory=dict)


@dataclass
class CloudSample:
    """
    Single training sample returned by any dataset adapter.

    Tensors:
        cloudy: [3, H, W] float32 in [0,1] — Green, Red, NIR
        clear:  [3, H, W] float32 in [0,1]
        mask:   [1, H, W] float32 binary (1 = cloud)

    All three tensors share the same spatial dimensions (patch_size × patch_size).
    """
    cloudy: torch.Tensor
    clear: torch.Tensor
    mask: torch.Tensor
    meta: CloudSceneMeta = field(default_factory=CloudSceneMeta)


class CloudRemovalDataset(ABC, Dataset):
    """
    Abstract base for all cloud-removal dataset adapters.

    Subclasses must implement:
        _collect_samples() → list of sample identifiers
        _load_sample(stem) → CloudSample

    Subclasses may override:
        _augment(sample) → CloudSample
    """

    BAND_NAMES: list[str] = ["Green", "Red", "NIR"]

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        patch_size: int = 256,
        augment: bool = True,
        min_cloud_fraction: float = 0.02,
        max_cloud_fraction: float = 1.0,
        min_pair_quality: float = 0.0,
    ) -> None:
        """
        Args:
            root_dir:            Root directory of the dataset.
            split:               'train', 'val', or 'test'.
            patch_size:          Output spatial patch size (pixels).
            augment:             Apply geometric augmentation (train only).
            min_cloud_fraction:  Skip samples with less cloud than this.
                                 Prevents near-clear samples from wasting capacity.
            max_cloud_fraction:  Skip samples with more cloud than this.
                                 Filters totally obscured scenes with no reference signal.
            min_pair_quality:    Minimum pair-quality score [0,1] from dataset auditing.
        """
        super().__init__()
        self.root = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.augment = augment and (split == "train")
        self.min_cloud_fraction = min_cloud_fraction
        self.max_cloud_fraction = max_cloud_fraction
        self.min_pair_quality = min_pair_quality

        self._stems: list[str] = []
        self._quality_scores: dict[str, float] = {}
        self._build()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _collect_samples(self) -> list[str]:
        """Return sorted list of sample stems present in all required sub-dirs."""
        ...

    @abstractmethod
    def _load_sample(self, stem: str) -> CloudSample:
        """Load a single sample by stem. Must return a CloudSample."""
        ...

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Human-readable dataset identifier (e.g. 'SEN12MS-CR')."""
        ...

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Collect and filter samples."""
        all_stems = self._collect_samples()

        # Apply quality score filtering if scores are pre-computed
        filtered = [
            s for s in all_stems
            if self._quality_scores.get(s, 1.0) >= self.min_pair_quality
        ]

        self._stems = filtered

        if not self._stems:
            raise RuntimeError(
                f"[{self.dataset_name}] No valid samples found in {self.root}. "
                f"Checked {len(all_stems)} candidates with min_pair_quality="
                f"{self.min_pair_quality}."
            )

    def load_quality_scores(self, scores_path: str | Path) -> None:
        """
        Load pre-computed pair quality scores from a JSON file.

        Produced by ai/dataset_tools/pair_quality.py.
        Expected format: {"stem": score, ...}

        Call before iterating to enable quality filtering.
        """
        import json
        with open(scores_path) as f:
            self._quality_scores = json.load(f)
        # Re-apply filter with loaded scores
        self._build()

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._stems)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | float]:
        stem = self._stems[idx]
        sample = self._load_sample(stem)

        # Cloud fraction filter — resample if out of range
        cf = sample.meta.cloud_fraction
        if self.split == "train" and not (self.min_cloud_fraction <= cf <= self.max_cloud_fraction):
            # Resample a random valid index (avoid infinite loops with a fallback)
            import random
            for _ in range(10):
                new_idx = random.randint(0, len(self) - 1)
                if new_idx != idx:
                    return self.__getitem__(new_idx)

        if self.augment:
            sample = self._augment(sample)

        return {
            "cloudy": sample.cloudy,
            "clear": sample.clear,
            "mask": sample.mask,
            "stem": sample.meta.stem,
            "cloud_fraction": sample.meta.cloud_fraction,
            "pair_quality": sample.meta.pair_quality_score,
            "dataset": self.dataset_name,
        }

    # ------------------------------------------------------------------
    # Augmentation (can be overridden)
    # ------------------------------------------------------------------

    def _augment(self, sample: CloudSample) -> CloudSample:
        """
        Geometric augmentation: random H/V flip + 90° rotation.

        These are radiometrically neutral — they do NOT alter spectral values,
        which is critical for satellite imagery.
        """
        import random

        c, cl, m = sample.cloudy.numpy(), sample.clear.numpy(), sample.mask.numpy()

        if random.random() > 0.5:
            c = np.flip(c, axis=2).copy()
            cl = np.flip(cl, axis=2).copy()
            m = np.flip(m, axis=2).copy()
        if random.random() > 0.5:
            c = np.flip(c, axis=1).copy()
            cl = np.flip(cl, axis=1).copy()
            m = np.flip(m, axis=1).copy()
        k = random.randint(0, 3)
        if k > 0:
            c = np.rot90(c, k, axes=(1, 2)).copy()
            cl = np.rot90(cl, k, axes=(1, 2)).copy()
            m = np.rot90(m, k, axes=(1, 2)).copy()

        return CloudSample(
            cloudy=torch.from_numpy(c),
            clear=torch.from_numpy(cl),
            mask=torch.from_numpy(m),
            meta=sample.meta,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _random_crop_chw(
        *arrays: np.ndarray,
        patch_size: int,
    ) -> tuple[np.ndarray, ...]:
        """Random spatial crop, input/output [C, H, W]."""
        import random
        H = arrays[0].shape[1]
        W = arrays[0].shape[2]
        p = patch_size
        if H < p or W < p:
            # Reflect-pad if smaller
            pad_h = max(0, p - H)
            pad_w = max(0, p - W)
            arrays = tuple(
                np.pad(a, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
                for a in arrays
            )
            H, W = arrays[0].shape[1], arrays[0].shape[2]
        top = random.randint(0, H - p)
        left = random.randint(0, W - p)
        return tuple(a[:, top:top + p, left:left + p] for a in arrays)
