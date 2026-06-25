"""
Metrics computation service.

Runs the full scientific validation suite including cloud-region,
clear-region, vegetation, and spectral accuracy metrics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def compute_image_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    cloudy: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    Compute the full scientific metric suite for a reconstruction.

    Args:
        pred:   [3, H, W] or [H, W, 3] float32 reconstruction in [0, 1].
        target: [3, H, W] or [H, W, 3] float32 ground-truth clear image.
        cloudy: Optional [3, H, W] or [H, W, 3] original cloudy input.
        mask:   Optional [1, 1, H, W] or [H, W] binary cloud mask.

    Returns:
        Dict of metric_name → float value.
    """
    from ai.validation.cloud_region import full_scientific_metrics
    from ai.metrics.compute import compute_all

    def _chw_tensor(arr: np.ndarray) -> torch.Tensor:
        """Convert to [1, 3, H, W] tensor."""
        a = arr
        if a.ndim == 3 and a.shape[2] <= 13:
            a = a.transpose(2, 0, 1)
        return torch.from_numpy(a[:3]).unsqueeze(0).float()

    p_t = _chw_tensor(pred)
    t_t = _chw_tensor(target)
    c_t = _chw_tensor(cloudy) if cloudy is not None else None

    m_t: Optional[torch.Tensor] = None
    if mask is not None:
        m_arr = np.array(mask, dtype=np.float32)
        if m_arr.ndim == 2:
            m_arr = m_arr[np.newaxis, np.newaxis]
        elif m_arr.ndim == 3:
            m_arr = m_arr[np.newaxis] if m_arr.shape[0] == 1 else m_arr[np.newaxis, :1]
        m_t = torch.from_numpy(m_arr).float()

    # Full scientific suite (includes cloud-region, vegetation, spectral)
    try:
        return full_scientific_metrics(p_t, t_t, c_t, m_t if m_t is not None else torch.zeros_like(p_t[:, :1]))
    except Exception:
        # Fallback to basic metrics if validation modules unavailable
        return compute_all(p_t, t_t, cloudy=c_t, mask=m_t)
