# Evaluation Protocol

## Objectives

Quantify cloud removal quality on the SEN2-CR test split and (where available)
on real LISS-IV paired scenes, using a standardised, reproducible protocol.

---

## Metrics

| Metric | Unit | Formula | Better |
|--------|------|---------|--------|
| **PSNR** | dB | 10·log₁₀(1/MSE) | Higher |
| **SSIM** | [0,1] | Structural similarity | Higher |
| **SAM** | rad or ° | arccos(p·t / ‖p‖‖t‖) | Lower |
| **RMSE** | [0,1] | √MSE | Lower |
| **NDVI MAE (global)** | [0,2] | mean|NDVI_pred−NDVI_gt| | Lower |
| **NDVI MAE (cloud)** | [0,2] | same, restricted to cloud mask | Lower |
| **NDVI Improvement** | [−2,2] | NDVI_MAE_baseline − NDVI_MAE_pred | Higher |

All image metrics are computed on the full reconstructed image (not just cloud regions)
unless otherwise specified.

---

## Evaluation Regions

### Region of Interest (ROI) Types

The demo dataset (`datasets/demo/`) contains three representative ROI types:

| ROI | Characteristics | Key Challenge |
|-----|-----------------|---------------|
| `agriculture/` | Farmland, seasonal crop cycle | NDVI accuracy critical |
| `urban/` | Mixed built-up / vegetation | Fine-scale texture recovery |
| `water/` | River, reservoir, coastal | Specular reflection, low NDVI |

---

## Standard Evaluation Run

### Step 1 — Inference

```bash
python -m ai.restormer.infer \
    --checkpoint ai/restormer/checkpoints/phase2_final.pt \
    --input datasets/sen2cr/cloudy/TEST_SCENE.npy \
    --mask  datasets/sen2cr/masks/TEST_SCENE.npy \
    --output outputs/predictions/TEST_SCENE.npy \
    --tile
```

### Step 2 — Batch Metrics

```bash
python -m ai.metrics.compute \
    --pred   outputs/predictions \
    --target datasets/sen2cr/clear \
    --mask   datasets/sen2cr/masks \
    --output outputs/metrics/report.json
```

### Step 3 — Inspect Report

```bash
cat outputs/metrics/report.json
```

---

## Baseline Comparisons

Evaluate against these baselines for ablation studies:

| Baseline | Description |
|----------|-------------|
| **Cloudy input** | Pass-through (no cloud removal) |
| **Temporal median** | Pixel-wise median of available clear scenes |
| **Pix2Pix** | Previous cGAN-based approach |
| **Restormer (no res. head)** | Backbone only, no cloud-adaptive head |
| **Restormer (full)** | Complete model (proposed) |

---

## Reporting Format

Report the following per experiment:

```json
{
  "model": "Restormer-v2",
  "dataset": "SEN2-CR test split",
  "n_samples": 200,
  "psnr_db": 32.4,
  "ssim": 0.917,
  "sam_deg": 4.76,
  "rmse": 0.031,
  "ndvi_mae_global": 0.042,
  "ndvi_mae_cloud": 0.061,
  "ndvi_improvement": 0.129
}
```

---

## Statistical Significance

- Report mean ± standard deviation over all test scenes.
- For NDVI metrics, report separately for cloud fractions:
  - Low:    0–30%
  - Medium: 30–60%
  - High:   60–100%

---

## Reference Targets (SEN2-CR literature)

| Metric | State-of-the-art range |
|--------|----------------------|
| PSNR   | 28–34 dB             |
| SSIM   | 0.88–0.94            |
| SAM    | 3°–7°                |
| RMSE   | 0.02–0.05            |

Models achieving PSNR > 30 dB and SSIM > 0.90 are considered competitive.
