# Known Limitations

## Model Limitations

### 1. Thick Cloud Occlusion
- Dense, opaque clouds remove all surface reflectance signal.
- The Restormer learns to hallucinate plausible surface texture from surrounding context.
- Reconstructed pixels under thick clouds are **estimates, not observations**.
- NDVI and other index-derived analytics in these regions carry higher uncertainty.

### 2. Cloud Shadow
- The current cloud mask targets cloud bodies, not shadow footprints.
- Shadow-obscured pixels may have lower reflectance but are not masked.
- Model performance degrades slightly in shadow regions compared to bright cloud areas.

### 3. Temporal Mismatch
- The model is trained on single-date reconstruction.
- Rapidly changing surfaces (floods, crops, fire scars) between the cloudy date
  and any temporal reference may introduce artefacts.

### 4. Band Coverage
- Only three bands are supported: Green, Red, NIR.
- SWIR bands (important for fire/moisture detection) are not included.
- Thermal infrared is not available.

---

## Domain Adaptation Limitations

### 5. Spectral Coefficients
- The linear gain/offset coefficients in `spectral_mapping.py` are representative values.
- They were not derived from co-located LISS-IV / Sentinel-2 scene pairs.
- For operational use, re-derive from at least 50 co-registered clear scenes.

### 6. Spatial Resolution Gap
- Sentinel-2 at 10 m is upsampled to simulate LISS-IV at 5.8 m.
- The unsharp-mask approach enhances existing edges but cannot recover truly
  sub-10 m detail absent from the Sentinel-2 source data.

---

## Inference Limitations

### 7. Tiled Inference Boundary Artefacts
- Hann-window blending minimises seam artefacts but does not eliminate them completely
  when the cloud mask has sharp boundaries near tile edges.
- Increasing overlap from 32 to 64+ pixels reduces this at the cost of more computation.

### 8. Memory and Speed
- A 512×512 image requires ~4 Restormer forward passes (tiled).
- A 4096×4096 ortho mosaic requires ~256 passes; expect ~2–5 minutes on a single GPU.
- CPU inference is supported but ~30× slower.

---

## Training Data Limitations

### 9. SEN2-CR Geographic Bias
- SEN2-CR scenes are predominantly from Europe and temperate regions.
- Performance on tropical (dense canopy, persistent deep convective cloud) or
  arid (bright bare soil) Indian scenes may be lower.
- Phase 3 LISS-IV fine-tuning is recommended with regionally representative data.

### 10. Synthetic Cloud Masks
- When real cloud masks are unavailable, `synthetic_masks.py` generates procedural
  cloud shapes.
- Synthetic masks do not accurately reproduce the radiometric properties of real
  clouds (e.g., cloud optical thickness variation, 3-D shadowing effects).

---

## Operational Constraints

### 11. No Geometric Correction
- The pipeline assumes the input image is already orthorectified and geo-referenced.
- No DEM-based terrain correction is performed.

### 12. Single-Date Input
- The model operates on a single cloudy scene.
- Multi-temporal compositing (`ai/temporal/compare.py`) is available but requires
  a manually supplied cloud-free reference image.

### 13. Checkpoint Availability
- No pre-trained weights are distributed in this repository.
- Users must train from scratch on SEN2-CR (or a compatible dataset) using
  `python -m ai.restormer.train`.
