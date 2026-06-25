# ChaturVyuha CloudVision AI

**Generative AI-Based Cloud Removal and Reconstruction for LISS-IV Satellite Imagery**

ISRO Hackathon · Problem Statement PS-3

---

## Architecture

| Component | Description |
|---|---|
| **Restormer** | Multi-Dconv Head Transposed Attention (MDTA) + Gated-Dconv FFN (GDFN) |
| **Cloud-Adaptive Residual Head** | Modulates features at cloud locations; clear pixels are never modified |
| **Loss** | L1 + SAM + MS-SSIM + Gradient + Adversarial |
| **Cloud Detector** | Lightweight U-Net segmentation model |
| **Domain Adaptation** | SEN2-CR B3/B4/B8 → LISS-IV Green/Red/NIR with histogram matching |
| **Tiled Inference** | 256×256 tiles, 32-pixel overlap, Hann-window blending |
| **Backend** | FastAPI — upload / detect / reconstruct / metrics / report endpoints |
| **Frontend** | React — drag-drop upload, before/after slider, metrics radar chart |

## Quick Start

### Training

```bash
pip install -r ai/requirements.txt
python -m ai.restormer.train --config ai/restormer/train_config.yaml
```

### Inference (single image)

```bash
python -m ai.restormer.infer \
    --checkpoint ai/restormer/checkpoints/phase2_final.pt \
    --input path/to/cloudy.npy \
    --mask  path/to/mask.npy \
    --output outputs/predictions/result.npy \
    --tile
```

### Backend API

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs

### Docker

```bash
cd docker
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend:  http://localhost:8000

---

## Sensor

**LISS-IV (Resourcesat-2 / Resourcesat-2A)**

| Band | Wavelength | Resolution |
|---|---|---|
| Green | ~560 nm | 5.8 m |
| Red   | ~665 nm | 5.8 m |
| NIR   | ~820 nm | 5.8 m |

---

## Dataset

SEN2-CR — Sentinel-2 Cloud Removal dataset.  
Band selection: B3 (Green) · B4 (Red) · B8 (NIR)

---

## Metrics

| Metric | Description |
|---|---|
| PSNR | Peak Signal-to-Noise Ratio (dB) |
| SSIM | Structural Similarity Index |
| SAM | Spectral Angle Mapper (radians / degrees) |
| RMSE | Root-Mean-Square Error |
| NDVI MAE | Mean Absolute Error of NDVI in cloud regions |

---

## License

For ISRO Hackathon evaluation purposes only.
