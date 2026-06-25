"""
Cloud detection service layer.

Exposes a thin, stateless interface over the CloudDetector model.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from backend.services.model_loader import get_cloud_detector, get_device


def detect_clouds(
    image: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, np.ndarray | float]:
    """
    Run cloud detection on a single image.

    Args:
        image:     [H, W, 3] float32 array in [0, 1] (Green, Red, NIR).
        threshold: Binarisation threshold for the probability map.

    Returns:
        Dictionary with:
            'probability_map': [H, W] float32 cloud probability ∈ [0, 1].
            'binary_mask':     [H, W] binary float32 mask (1 = cloud).
            'cloud_fraction':  Float cloud cover fraction.
    """
    detector = get_cloud_detector()
    device = get_device()

    if detector is None:
        # Fall back to a simple threshold heuristic when model is not loaded
        return _heuristic_detection(image, threshold)

    tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # The U-Net detector pools by 2 three times; its skip-connection concats
    # require H and W to be multiples of 8, otherwise the decoder tensors are
    # off-by-one and torch.cat raises a shape-mismatch error. Reflect-pad to the
    # next multiple of 8, then crop the probability map back to the true size.
    _, _, H, W = tensor.shape
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        prob = detector(tensor)[..., :H, :W]
        prob_map = prob.squeeze(0).squeeze(0).cpu().numpy()

    binary = (prob_map > threshold).astype(np.float32)
    return {
        "probability_map": prob_map,
        "binary_mask": binary,
        "cloud_fraction": float(binary.mean()),
    }


def _heuristic_detection(
    image: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, np.ndarray | float]:
    """
    Brightness-based cloud heuristic used as fallback when the model is absent.

    High reflectance in all three bands typically indicates clouds.
    """
    brightness = image.mean(axis=-1)
    prob = np.clip((brightness - 0.3) / 0.5, 0.0, 1.0).astype(np.float32)
    binary = (prob > threshold).astype(np.float32)
    return {
        "probability_map": prob,
        "binary_mask": binary,
        "cloud_fraction": float(binary.mean()),
    }
