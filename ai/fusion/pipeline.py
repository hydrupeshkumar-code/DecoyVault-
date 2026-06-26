"""
Full cloud removal inference pipeline combining cloud detection,
Restormer reconstruction, and confidence fusion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ai.cloud_detector.model import CloudDetector
from ai.fusion.fuse import confidence_weighted_fuse
from ai.restormer.model import Restormer
from ai.restormer.tiled_inference import tiled_infer


class CloudRemovalPipeline:
    """
    End-to-end cloud removal for a single LISS-IV image.

    Components:
        1. Cloud detector  → binary mask
        2. Restormer       → cloud-free reconstruction
        3. Confidence fusion → final output

    Args:
        restormer_ckpt:      Path to Restormer checkpoint.
        detector_ckpt:       Path to CloudDetector checkpoint (optional).
        device:              Torch device string or torch.device.
        tile_size:           Tile size for tiled inference.
        overlap:             Tile overlap.
        cloud_threshold:     Binarisation threshold for cloud probability map.
    """

    def __init__(
        self,
        restormer_ckpt: str | Path,
        detector_ckpt: Optional[str | Path] = None,
        device: str | torch.device = "auto",
        tile_size: int = 256,
        overlap: int = 32,
        cloud_threshold: float = 0.5,
    ) -> None:
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tile_size = tile_size
        self.overlap = overlap
        self.cloud_threshold = cloud_threshold

        # Restormer
        self.restormer = Restormer().to(self.device)
        state = torch.load(restormer_ckpt, map_location=self.device, weights_only=False)
        if "model_state" in state:
            state = state["model_state"]
        self.restormer.load_state_dict(state)
        self.restormer.eval()

        # Cloud detector (optional)
        self.detector: Optional[CloudDetector] = None
        if detector_ckpt and Path(detector_ckpt).exists():
            self.detector = CloudDetector(in_channels=3).to(self.device)
            self.detector.load_state_dict(
                torch.load(detector_ckpt, map_location=self.device, weights_only=False)
            )
            self.detector.eval()

    @torch.no_grad()
    def run(
        self,
        image: np.ndarray,
        external_mask: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """
        Run the full pipeline on a single image.

        Args:
            image:         [H, W, 3] float32 array in [0, 1].
            external_mask: Optional pre-computed binary cloud mask [H, W].
                           If None and a detector is loaded, the detector
                           is used.  If no detector, assume full-image cloud.

        Returns:
            Dictionary with keys:
                'output':      [H, W, 3] cloud-removed image.
                'cloud_mask':  [H, W] binary cloud mask used.
                'confidence':  [H, W] model confidence map (if available).
        """
        H, W, _ = image.shape

        # Step 1: Determine cloud mask
        if external_mask is not None:
            mask_np = (external_mask > 0.5).astype(np.float32)
        elif self.detector is not None:
            t = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            prob = self.detector(t).squeeze(0).squeeze(0).cpu().numpy()
            mask_np = (prob > self.cloud_threshold).astype(np.float32)
        else:
            mask_np = np.ones((H, W), dtype=np.float32)

        # Step 2: Restormer reconstruction (tiled)
        reconstructed = tiled_infer(
            self.restormer, image, mask_np, self.device,
            tile_size=self.tile_size, overlap=self.overlap,
        )

        # Step 3: Confidence fusion
        img_t = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        rec_t = torch.from_numpy(reconstructed.transpose(2, 0, 1)).unsqueeze(0)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

        fused = confidence_weighted_fuse(img_t, rec_t, mask_t).squeeze(0)
        fused_np = fused.cpu().numpy().transpose(1, 2, 0)

        return {
            "output": np.clip(fused_np, 0.0, 1.0).astype(np.float32),
            "cloud_mask": mask_np,
            "reconstructed": reconstructed,
        }
