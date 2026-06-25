# SEN2-CR Data Pipeline

## Dataset Overview

**SEN2-CR** (Sentinel-2 Cloud Removal) is a large-scale multi-temporal dataset
providing paired cloudy/cloud-free Sentinel-2 imagery with cloud masks.

| Property | Value |
|----------|-------|
| Sensor | Sentinel-2 MSI |
| Bands | 13 (B1–B12 + B8A) |
| Spatial resolution | 10 m (B2/B3/B4/B8), 20 m (others), 60 m (B1/B9/B10) |
| Patch size | Variable (typically 256×256 to 512×512 pixels) |
| Cloud fraction | 0–100% per patch |
| Format | GeoTIFF or NumPy `.npy` |

---

## Folder Layout

```
datasets/sen2cr/
├── cloudy/    # Sentinel-2 images with cloud contamination
├── clear/     # Cloud-free reference images (same date range)
└── masks/     # Binary cloud masks (1 = cloud, 0 = clear)
```

File naming: matching stems across all three directories.
Example: `ROI_0001.npy` in cloudy/, clear/, and masks/.

---

## Preprocessing Steps

### 1. Band Selection

From 13 SEN2-CR bands, select indices [2, 3, 7] (0-indexed):

```python
SEN2_BAND_INDICES = [2, 3, 7]   # B3=Green, B4=Red, B8=NIR
```

### 2. Normalisation

All reflectance values are normalised to [0, 1].
SEN2-CR typically stores values as TOA reflectance × 10000 (uint16).
Divide by 10000 and clip to [0, 1].

```python
image = np.clip(image / 10000.0, 0.0, 1.0)
```

### 3. Spectral Mapping

Apply linear gain/offset to simulate LISS-IV spectral response:

```python
image = sen2_to_liss4_spectral(image)   # ai/domain_adaptation/spectral_mapping.py
```

### 4. Patch Extraction

Random 256×256 crops during training with:
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random 90° rotation (k ∈ {0,1,2,3})

### 5. LISS-IV Simulation Augmentations

Applied during Phase 3 fine-tuning:
- Gaussian noise (std=0.005, p=0.5)
- Radiometric jitter (gain ∈ [0.95,1.05], bias ∈ [−0.02,0.02], p=0.5)
- Unsharp mask sharpening (strength=0.2, p=0.4)
- Cloud texture randomisation

---

## DataLoader Configuration

```yaml
data:
  root_dir: "datasets/sen2cr"
  patch_size: 256
  num_workers: 4
  pin_memory: true
```

Train/val split: 90% / 10% (random split per run).

---

## Cloud Mask Convention

Binary masks:
- `1.0` = cloud pixel
- `0.0` = clear pixel

Thick clouds, thin clouds, and cloud shadows are all labelled as `1`.

---

## Dataset Statistics (typical SEN2-CR subset)

| Split | Scenes | Mean Cloud Cover |
|-------|--------|-----------------|
| Train | ~2,000 | 38% |
| Val   | ~200   | 36% |
| Test  | ~200   | 40% |

---

## Extending to Real LISS-IV Data

When real LISS-IV paired data is available, place it under `datasets/liss4/`
with the same `cloudy/`, `clear/`, `masks/` structure and point Phase 3 to it:

```yaml
phase3:
  liss4_root: "datasets/liss4"
```

The `SEN2CRDataset` loader is compatible with LISS-IV `.npy` files that are
already 3-band (no band-selection step is applied when `arr.shape[-1] == 3`).
