"""Dataset loader for cloud detection training."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CloudDetectionDataset(Dataset):
    """
    Dataset for cloud detection (binary segmentation).

    Expects:
        root_dir/images/  *.npy  [H, W, 3] Green/Red/NIR
        root_dir/masks/   *.npy  [H, W]    binary cloud mask

    Args:
        root_dir:   Root directory containing 'images' and 'masks'.
        patch_size: Spatial crop.
        augment:    Apply flips and rotations.
    """

    def __init__(self, root_dir: str, patch_size: int = 256, augment: bool = True) -> None:
        super().__init__()
        self.images_dir = Path(root_dir) / "images"
        self.masks_dir = Path(root_dir) / "masks"
        self.patch_size = patch_size
        self.augment = augment

        img_stems = {p.stem for p in self.images_dir.glob("*.npy")}
        msk_stems = {p.stem for p in self.masks_dir.glob("*.npy")}
        self.stems = sorted(img_stems & msk_stems)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        stem = self.stems[idx]
        img = np.load(str(self.images_dir / f"{stem}.npy")).astype(np.float32)
        msk = np.load(str(self.masks_dir / f"{stem}.npy")).astype(np.float32)
        msk = (msk > 0.5).astype(np.float32)

        # Crop
        H, W = img.shape[:2]
        p = self.patch_size
        if H > p and W > p:
            y = random.randint(0, H - p)
            x = random.randint(0, W - p)
            img = img[y : y + p, x : x + p]
            msk = msk[y : y + p, x : x + p]

        if self.augment:
            if random.random() > 0.5:
                img, msk = np.fliplr(img).copy(), np.fliplr(msk).copy()
            if random.random() > 0.5:
                img, msk = np.flipud(img).copy(), np.flipud(msk).copy()

        img_t = torch.from_numpy(np.clip(img, 0, 1).transpose(2, 0, 1))
        msk_t = torch.from_numpy(msk).unsqueeze(0)
        return {"image": img_t, "mask": msk_t}
