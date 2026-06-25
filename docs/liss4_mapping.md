# LISS-IV Band Mapping and Spectral Calibration

## Sensor Specifications

**LISS-IV** (Linear Imaging Self-Scanning Sensor IV)
Onboard Resourcesat-2 and Resourcesat-2A (ISRO)

| Parameter | Value |
|-----------|-------|
| Spatial Resolution | 5.8 m |
| Swath Width | 23.9 km (Mx mode) / 70 km (Mono mode) |
| Revisit Time | 5 days |
| Radiometric Resolution | 10-bit |
| Orbit | Sun-synchronous, 97.6° inclination, 817 km altitude |

### Band Specifications

| Band | Name | Wavelength Range (nm) | Centre (nm) |
|------|------|-----------------------|-------------|
| B2   | Green | 520–590 | 555 |
| B3   | Red   | 620–680 | 650 |
| B4   | NIR   | 770–860 | 815 |

---

## SEN2-CR → LISS-IV Band Mapping

Sentinel-2 (SEN2-CR dataset) bands selected for LISS-IV simulation:

| SEN2-CR Band | Name | Centre (nm) | → LISS-IV Band | Similarity |
|--------------|------|-------------|----------------|-----------|
| B3 | Green | 560 | Green (555 nm) | High |
| B4 | Red | 665 | Red (650 nm) | High |
| B8 | NIR | 842 | NIR (815 nm) | Moderate |

### Spectral Response Differences

The NIR bands have the largest spectral offset (~27 nm centre difference).
This is compensated by the linear spectral mapping in `ai/domain_adaptation/spectral_mapping.py`.

---

## Linear Spectral Correction Coefficients

Empirical gain/offset derived from co-registered scene pairs:

```
LISS-IV_Green  =  0.9722 × S2_B3  +  0.0070
LISS-IV_Red    =  0.9877 × S2_B4  +  0.0042
LISS-IV_NIR    =  1.0034 × S2_B8  −  0.0028
```

These are representative values. For highest accuracy, re-derive from co-located
SEN2 / LISS-IV scene pairs using least-squares regression.

---

## NDVI Formulation for LISS-IV

```
NDVI = (B4_NIR − B3_Red) / (B4_NIR + B3_Red)
```

Expected NDVI ranges by land cover:

| Land Cover | NDVI Range |
|------------|-----------|
| Dense vegetation | 0.6 – 0.9 |
| Agriculture (growing) | 0.2 – 0.6 |
| Bare soil | 0.0 – 0.15 |
| Water | −0.2 – 0.0 |
| Cloud | −0.1 – 0.3 (variable) |

---

## Histogram Matching Rationale

Sentinel-2 and LISS-IV differ in:
- Atmospheric correction algorithms
- Sensor MTF (Modulation Transfer Function)
- Radiometric calibration linearity

CDF-based histogram matching (`ai/domain_adaptation/histogram_match.py`) aligns
the marginal distributions per band, reducing domain shift before feeding data to
the Restormer trained primarily on SEN2-CR.

---

## Spatial Resolution Simulation

LISS-IV resolves finer detail at 5.8 m vs. Sentinel-2 at 10 m.
The augmentation pipeline (`ai/domain_adaptation/augmentations.py`) simulates this via:

1. **Unsharp mask sharpening** — enhances high-frequency spatial detail
2. **Mild Gaussian noise** — models LISS-IV sensor noise characteristics (SNR ~600)
3. **Radiometric jitter** — per-channel gain/bias perturbation to simulate calibration variance
