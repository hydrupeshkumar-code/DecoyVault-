"""
Loss functions for Restormer cloud removal training.

Total Loss = 1.0 × L1  +  0.5 × SAM  +  0.3 × MS-SSIM  +  0.2 × Gradient  +  0.05 × Adversarial
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SAM Loss  (Spectral Angle Mapper)
# ---------------------------------------------------------------------------

class SAMLoss(nn.Module):
    """
    Spectral Angle Mapper loss.

    Computes the mean angular distance between predicted and target spectra
    across the channel dimension.  Output ∈ [0, π/2].

    Args:
        eps: Small value to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, C, H, W]
            target: [B, C, H, W]

        Returns:
            Scalar SAM loss.
        """
        dot = (pred * target).sum(dim=1)                  # [B, H, W]
        # CRITICAL: compute the norm as sqrt(sum_of_squares + eps), NOT as
        # tensor.norm().clamp(min=eps). torch.norm has a 0/0 NaN *gradient* at
        # all-zero (black) pixels — clamping the forward OUTPUT does not fix the
        # backward pass. Putting eps INSIDE the sqrt makes the gradient finite
        # everywhere. Black pixels are common in real data (image borders,
        # nodata fill), so this is the difference between training and NaN.
        norm_pred = torch.sqrt((pred * pred).sum(dim=1) + self.eps)
        norm_tgt = torch.sqrt((target * target).sum(dim=1) + self.eps)
        # Keep cos strictly inside (-1, 1): acos has infinite gradient at the
        # endpoints. A 1e-6 margin caps the gradient at ~1/sqrt(2e-6) ≈ 700.
        cos_angle = (dot / (norm_pred * norm_tgt)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        angle = torch.acos(cos_angle)                     # [B, H, W]
        return angle.mean()


# ---------------------------------------------------------------------------
# MS-SSIM Loss
# ---------------------------------------------------------------------------

def _gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """1-D Gaussian kernel."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel: torch.Tensor,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Single-scale SSIM helper. Returns map [B, C, H, W]."""
    B, C, H, W = pred.shape
    k = kernel.view(1, 1, -1, 1) * kernel.view(1, 1, 1, -1)
    k = k.expand(C, 1, -1, -1).to(pred.device)
    pad = kernel.shape[0] // 2

    mu_x = F.conv2d(pred, k, padding=pad, groups=C)
    mu_y = F.conv2d(target, k, padding=pad, groups=C)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, k, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(target * target, k, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(pred * target, k, padding=pad, groups=C) - mu_xy

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return numerator / denominator.clamp(min=1e-8)


class MSSSIMLoss(nn.Module):
    """
    Multi-Scale Structural Similarity loss.

    Args:
        num_scales: Number of pyramid levels.
        kernel_size: Gaussian kernel size.
        sigma: Gaussian standard deviation.
        weights: Per-scale weights (must sum to 1).
    """

    _DEFAULT_WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    def __init__(
        self,
        num_scales: int = 5,
        kernel_size: int = 11,
        sigma: float = 1.5,
        weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.register_buffer("kernel", _gaussian_kernel(kernel_size, sigma))
        self.weights = weights if weights is not None else self._DEFAULT_WEIGHTS[:num_scales]
        assert len(self.weights) == num_scales

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, C, H, W] ∈ [0, 1]
            target: [B, C, H, W] ∈ [0, 1]

        Returns:
            Scalar MS-SSIM loss ∈ [0, 1].
        """
        kernel = self.kernel  # type: ignore[attr-defined]
        mcs_values: list[torch.Tensor] = []

        x, y = pred, target
        for scale in range(self.num_scales):
            ssim_map = _ssim(x, y, kernel)
            if scale < self.num_scales - 1:
                mcs_values.append(ssim_map.mean())
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
                y = F.avg_pool2d(y, kernel_size=2, stride=2)
            else:
                mcs_values.append(ssim_map.mean())

        # Per-scale SSIM means can be slightly negative (especially early in
        # training when predictions are far from the target). Raising a negative
        # base to a fractional power yields NaN, which would poison the whole
        # loss. Clamp to a small positive floor before the weighted product.
        ms_ssim = torch.stack(
            [v.clamp(min=1e-6) ** w for v, w in zip(mcs_values, self.weights)]
        ).prod()
        return 1.0 - ms_ssim


# ---------------------------------------------------------------------------
# Gradient Loss
# ---------------------------------------------------------------------------

class GradientLoss(nn.Module):
    """
    Edge-preserving gradient loss (L1 on Sobel gradients).

    Penalises differences in spatial gradients, encouraging the model to
    preserve sharpness at cloud-free / cloud boundaries.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, C, H, W]
            target: [B, C, H, W]

        Returns:
            Scalar gradient loss.
        """
        pred_grad = self._gradients(pred)
        target_grad = self._gradients(target)
        return F.l1_loss(pred_grad, target_grad)

    @staticmethod
    def _gradients(x: torch.Tensor) -> torch.Tensor:
        """Compute horizontal + vertical Sobel gradients, concatenated on C."""
        B, C, H, W = x.shape
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=x.dtype, device=x.device,
        ).view(1, 1, 3, 3).expand(C, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=x.dtype, device=x.device,
        ).view(1, 1, 3, 3).expand(C, 1, 3, 3)
        gx = F.conv2d(x, sobel_x, padding=1, groups=C)
        gy = F.conv2d(x, sobel_y, padding=1, groups=C)
        return torch.cat([gx, gy], dim=1)


# ---------------------------------------------------------------------------
# Adversarial Loss (PatchGAN discriminator wrapper)
# ---------------------------------------------------------------------------

class AdversarialLoss(nn.Module):
    """
    Least-squares adversarial loss for the generator side.

    The discriminator is expected to be passed in externally and updated
    by the training loop.  This module only holds the generator's adversarial
    objective.
    """

    def forward(self, discriminator: nn.Module, fake: torch.Tensor) -> torch.Tensor:
        """
        Args:
            discriminator: PatchGAN-style discriminator network.
            fake:          Generator output [B, C, H, W].

        Returns:
            Generator adversarial loss scalar.
        """
        pred_fake = discriminator(fake)
        return 0.5 * torch.mean((pred_fake - 1.0) ** 2)


class PatchDiscriminator(nn.Module):
    """
    70×70 PatchGAN discriminator for cloud removal adversarial training.

    Args:
        in_channels: Input + target concatenated channels (6 for 3+3).
        ndf: Base feature count.
    """

    def __init__(self, in_channels: int = 6, ndf: int = 64) -> None:
        super().__init__()

        def block(ic: int, oc: int, stride: int, norm: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [
                nn.Conv2d(ic, oc, kernel_size=4, stride=stride, padding=1, bias=not norm)
            ]
            if norm:
                layers.append(nn.BatchNorm2d(oc))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, ndf, stride=2, norm=False),
            *block(ndf, ndf * 2, stride=2),
            *block(ndf * 2, ndf * 4, stride=2),
            *block(ndf * 4, ndf * 8, stride=1),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# Combined Loss
# ---------------------------------------------------------------------------

class CloudRemovalLoss(nn.Module):
    """
    Combined loss: L1 + SAM + MS-SSIM + Gradient + (optional) Adversarial.

    Weights:
        l1:          1.0
        sam:         0.5
        ms_ssim:     0.3
        gradient:    0.2
        adversarial: 0.05

    Args:
        use_adversarial: Include adversarial term (requires discriminator arg in forward).
    """

    def __init__(self, use_adversarial: bool = False) -> None:
        super().__init__()
        self.use_adversarial = use_adversarial
        self.sam_loss = SAMLoss()
        self.msssim_loss = MSSSIMLoss()
        self.gradient_loss = GradientLoss()
        if use_adversarial:
            self.adv_loss = AdversarialLoss()

        self.w_l1 = 1.0
        self.w_sam = 0.5
        self.w_msssim = 0.3
        self.w_gradient = 0.2
        self.w_adv = 0.05

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        discriminator: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred:          [B, 3, H, W] model output.
            target:        [B, 3, H, W] ground-truth clear image.
            discriminator: Optional PatchDiscriminator (required if use_adversarial).

        Returns:
            Dictionary with keys 'total', 'l1', 'sam', 'ms_ssim', 'gradient',
            and optionally 'adversarial'.
        """
        l1 = F.l1_loss(pred, target)
        sam = self.sam_loss(pred, target)
        ms_ssim = self.msssim_loss(pred, target)
        gradient = self.gradient_loss(pred, target)

        total = (
            self.w_l1 * l1
            + self.w_sam * sam
            + self.w_msssim * ms_ssim
            + self.w_gradient * gradient
        )

        losses = {
            "l1": l1,
            "sam": sam,
            "ms_ssim": ms_ssim,
            "gradient": gradient,
        }

        if self.use_adversarial and discriminator is not None:
            adv = self.adv_loss(discriminator, pred)
            total = total + self.w_adv * adv
            losses["adversarial"] = adv

        losses["total"] = total
        return losses
