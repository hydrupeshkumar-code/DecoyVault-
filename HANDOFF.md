# Local Handoff — CloudVision AI

Get from a fresh clone to **actually removing clouds** on your own machine.
Every command is runnable as-is from the repo root.

## Current state (what's already fixed)

- **Training import bug is fixed.** The `ai.geospatial` package was misplaced;
  it now lives at `ai/geospatial/`, so `python -m ai.restormer.train` no longer
  crashes with `ModuleNotFoundError: No module named 'ai.geospatial'`.
- **Smoke test runs with zero setup.** If no dataset is present, `--smoke`
  auto-generates a tiny synthetic one so you can prove training works offline.
- **What is NOT fixed (needs you):** there are **no trained weights** in the repo.
  The model has a global residual (`output(feats) + x`), so an *untrained* model
  outputs ≈ its input → the "clouds not removed" passthrough. Removing real clouds
  requires a real training run on a **GPU** (steps 3–4 below).

---

## 0. Setup (once)

```bash
git clone <your DecoyVault- repo URL>
cd DecoyVault-
git checkout claude/ponytail-plugin-install-ywtbg6   # branch with the fixes

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r ai/requirements.txt -r backend/requirements.txt
```

If `torch` fails to install, install the build for your machine from
https://pytorch.org (pick CPU or your CUDA version), then re-run the line above.

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```
`True` = you can do real training (steps 3–4). `False` = CPU only; the smoke
test (step 1) still works, full training does not.

---

## 1. Prove the pipeline works — no data, no GPU (~1 min)

```bash
python -m ai.restormer.train --smoke
```

Expected: it logs `generating synthetic smoke data`, the **mean loss drops each
epoch** (~0.30 → ~0.14), validation runs, and it writes
`ai/restormer/checkpoints/phase1_final.pt`. If this finishes, the model + loss +
training loop are sound — anything else wrong is data/weights, not code.

> This checkpoint is a sanity artifact only (2 epochs on toy data). It will NOT
> remove real clouds. Delete it before real training: `rm ai/restormer/checkpoints/*.pt`

---

## 2. Get real data

| Dataset | Role | Get it |
|---|---|---|
| RICE2 | quick smoke (RGB) | `bash download_datasets.sh` |
| **SEN12MS-CR** | **primary training** | https://mediatum.ub.tum.de/1554803 |
| Bhoonidhi LISS-IV | Phase 3 fine-tune (optional) | https://bhoonidhi.nrsc.gov.in/ |

SEN12MS-CR is huge — grab a **subset** (a few ROI folders); 2,000–5,000
quality pairs is plenty for a hackathon.

Arrange it into the flat layout the loader expects (`cloudy/ clear/ masks/`):

```bash
python -m ai.dataset_tools.prepare_dataset \
    --cloudy-dir datasets/raw/SEN12MSCR/s2_cloudy \
    --clear-dir  datasets/raw/SEN12MSCR/s2_cloudfree \
    --output     datasets/sen12ms_cr \
    --bands 2,3,7 --normalize scale --scale 10000 \
    --match order --max 4000
```

The default `ai/restormer/train_config.yaml` already points `root_dir` at
`datasets/sen12ms_cr` — no config edit needed.

---

## 3. Train for real (GPU)

```bash
python -m ai.restormer.train --config ai/restormer/train_config.yaml
# watch it:  tensorboard --logdir outputs/metrics
```

- Phase 1 (warmup) → Phase 2 (main). Phase 3 runs only if `datasets/liss4/` exists.
- Watch **validation cloud-region PSNR**; aim for ≥28 dB.
- Training writes `ai/restormer/checkpoints/phase2_final.pt` — the exact path the
  demo and backend look for.

Cap a quick first run to confirm the loss descends on real data before committing hours:
```bash
python -m ai.restormer.train --config ai/restormer/train_config.yaml --max-samples 200
```

---

## 4. See cloud removal

**Demo (4-panel PNG like the screenshot):**
```bash
python -m ai.demo_liss4 \
    --checkpoint ai/restormer/checkpoints/phase2_final.pt \
    --chips     datasets/liss4 \
    --output    demo_result.png
```
With a *trained* checkpoint the "Cloud-Removed" panel differs from the input and
the Change Map lights up over the cloud mask. With an untrained one it's a
passthrough — that's the symptom, not a bug.

**Backend + frontend:**
```bash
export RESTORMER_CHECKPOINT=ai/restormer/checkpoints/phase2_final.pt
uvicorn backend.main:app --host 0.0.0.0 --port 8000     # http://localhost:8000/docs
```
No checkpoint on disk → the backend logs a warning and `/reconstruct` returns
`fallback:true` (a flat mean-fill, not real removal). Point
`RESTORMER_CHECKPOINT` at your trained file to switch on the real model.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: ai.geospatial` | You're on old code — use the fixed branch (step 0). |
| `No module named 'numpy'` from torch | `pip install numpy` (install order issue). |
| Cloud-Removed looks identical to input | Untrained/undertrained checkpoint — train more (step 3). |
| Training exits: "Need at least 2 samples" | `root_dir` empty — run step 2 (`prepare_dataset`). |
| `.tif` read errors | `rasterio` missing/broken: `pip install rasterio`. |
| GPU OOM | Lower `phase*.batch_size` (2→1) or `data.patch_size` (128→64) in the config. |
| Loss goes NaN | Lower `phase1.lr`; NaN guards are already wired in. |
