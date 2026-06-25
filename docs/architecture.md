# System Architecture — ChaturVyuha CloudVision AI

## Overview

ChaturVyuha CloudVision AI is a production-ready cloud removal pipeline for
LISS-IV satellite imagery (Resourcesat-2/2A, 5.8 m GSD, Green/Red/NIR bands).

```
                        ┌──────────────────────────────────────┐
                        │          Input: LISS-IV Image         │
                        │          [H × W × 3]  (G/R/NIR)      │
                        └──────────────┬───────────────────────┘
                                       │
                     ┌─────────────────▼──────────────────┐
                     │         Cloud Detector (U-Net)      │
                     │   Binary segmentation → mask [H×W]  │
                     └─────────────────┬──────────────────┘
                                       │  cloud_mask
                     ┌─────────────────▼──────────────────┐
                     │        Restormer Backbone            │
                     │  MDTA + GDFN  ·  U-Net enc/dec       │
                     │  Input: [B,3,256,256]                │
                     └─────────────────┬──────────────────┘
                                       │  features
                     ┌─────────────────▼──────────────────┐
                     │   Cloud-Adaptive Residual Head       │
                     │  features × (1 + α·mask)  →  Δ      │
                     │  output = input + Δ · mask           │
                     └─────────────────┬──────────────────┘
                                       │
                     ┌─────────────────▼──────────────────┐
                     │       Confidence Fusion              │
                     │  Clear pixels ← original input       │
                     │  Cloud pixels ← Restormer output     │
                     └─────────────────┬──────────────────┘
                                       │
                        ┌──────────────▼───────────────────┐
                        │     Output: Reconstructed Image   │
                        │     [H × W × 3]  ∈ [0, 1]        │
                        └──────────────────────────────────┘
```

---

## Components

### 1. Cloud Detector (`ai/cloud_detector/`)

- **Architecture**: U-Net with double-conv blocks, 3-level encoder/decoder
- **Input**: `[B, 3, H, W]` Green/Red/NIR
- **Output**: `[B, 1, H, W]` probability map → binarised at threshold (default 0.5)
- **Training**: BCE loss, AdamW, CosineAnnealingLR

### 2. Restormer Backbone (`ai/restormer/model.py`)

- **MDTA** — Multi-Dconv Head Transposed Attention (channel-wise, O(C²) complexity)
- **GDFN** — Gated Depth-wise Feed-Forward Network with GELU gating
- **Scales**: 3 encoder levels + bottleneck; base dim = 48, heads = [1,2,4,8]
- **Skip connections** at each scale
- **Global residual**: output = decoder_out + input

### 3. Cloud-Adaptive Residual Head (`ai/restormer/residual_head.py`)

```
features_scaled = features × (1 + α × cloud_mask)
residual        = Conv(features_scaled)                  # ∈ ℝ³
output          = cloudy_input + residual × cloud_mask
```

`α` is a learned scalar. Clear-pixel identity is enforced by the mask multiplication.

### 4. Loss Function (`ai/restormer/losses.py`)

```
L_total = 1.0·L1  +  0.5·SAM  +  0.3·MS-SSIM  +  0.2·Gradient  +  0.05·Adversarial
```

### 5. Domain Adaptation (`ai/domain_adaptation/`)

SEN2-CR → LISS-IV mapping:

| SEN2 Band | Wavelength | LISS-IV Channel |
|-----------|-----------|-----------------|
| B3        | ~560 nm   | Green           |
| B4        | ~665 nm   | Red             |
| B8        | ~842 nm   | NIR             |

Pipeline: band selection → spectral gain/offset → histogram matching → normalisation

### 6. Tiled Inference (`ai/restormer/tiled_inference.py`)

- Tile size: 256 × 256
- Overlap: 32 pixels
- Blending: 2-D Hann window (eliminates seams)
- Supports images of arbitrary size

---

## Backend API

FastAPI application with five route groups:

| Route | Method | Description |
|-------|--------|-------------|
| `/upload/` | POST | Accept `.npy` image → returns `file_id` |
| `/detect/` | POST | Run cloud segmentation → returns `mask_id` |
| `/reconstruct/` | POST | Restormer inference → returns `result_id` |
| `/metrics/` | POST | PSNR/SSIM/SAM/RMSE/NDVI-MAE computation |
| `/report/generate` | POST | PDF/JSON quality report generation |

---

## Frontend

React SPA with:
- Drag-and-drop `.npy` upload
- Pipeline action buttons with live status
- Before/after comparison slider
- Metrics radar chart (Recharts)

---

## Training Phases

| Phase | Epochs | LR | Data | Notes |
|-------|--------|----|------|-------|
| 1 – Warmup | 20 | 1e-4 | SEN2-CR | Low LR, stabilise gradients |
| 2 – Main | 150 | 3e-4 | SEN2-CR | Full schedule + EMA |
| 3 – Fine-tune | 50 | 5e-5 | LISS-IV | Domain adaptation + adversarial |
