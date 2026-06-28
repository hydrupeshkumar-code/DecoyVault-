# Training Guide — ChaturVyuha CloudVision AI

End-to-end path from a fresh clone to a trained Restormer cloud-removal model
that the demo/backend picks up automatically. Every command below is runnable
as-is; the dataset-prep and `--smoke` tooling exist specifically so there are
**no blockers** between download and training.

> **GPU required for real training.** Phases 1–2 are infeasible on CPU. Use a
> CUDA machine (Colab/Kaggle T4, a cloud A100, or a local GPU). The `--smoke`
> sanity run works on CPU.

---

## 0. Environment

```bash
git clone <your DecoyVault- repo URL>
cd DecoyVault-
git checkout claude/zealous-rubin-b84awv

python -m venv .venv && source .venv/bin/activate
pip install -r ai/requirements.txt -r backend/requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # True on your GPU box
```

## 1. Get the datasets

| Dataset | Role | Bands | Get it |
|---|---|---|---|
| **RICE2** | Smoke test (ships masks, downloads fast) | RGB | `bash download_datasets.sh` |
| **SEN12MS-CR** | **Primary Phase 1+2** training | S2 13-band → G/R/NIR | [MediaTUM 1554803](https://mediatum.ub.tum.de/1554803) |
| **Bhoonidhi LISS-IV** | Phase 3 fine-tune (ISRO differentiator) | G/R/NIR | [bhoonidhi.nrsc.gov.in](https://bhoonidhi.nrsc.gov.in/) |

```bash
bash download_datasets.sh      # scaffolds datasets/, fetches RICE, prints next steps
```

SEN12MS-CR is huge (~100+ GB full). For a hackathon, grab a **subset** (a few ROI
folders) and cap with `--max` below — 2,000–5,000 quality-filtered pairs trains
in hours, not days.

## 2. Prepare (arrange + generate masks)

SEN12MS-CR ships cloudy + clear tiles but **no cloud masks**, and the loader
requires a `masks/` folder. `prepare_dataset.py` normalises any source into the
flat layout both loaders expect and derives masks from the cloudy-vs-clear
brightness difference (the model never consumes masks during training — they
only drive cloud-fraction filtering and metric reporting, so this is sufficient).

```bash
# SEN12MS-CR: slice Sentinel-2 bands B3/B4/B8 -> Green/Red/NIR, DN/10000 -> [0,1]
python -m ai.dataset_tools.prepare_dataset \
    --cloudy-dir datasets/raw/SEN12MSCR/s2_cloudy \
    --clear-dir  datasets/raw/SEN12MSCR/s2_cloudfree \
    --output     datasets/sen12ms_cr \
    --bands 2,3,7 --normalize scale --scale 10000 \
    --match order --max 4000
```

Output: `datasets/sen12ms_cr/{cloudy,clear,masks}/pair_000001.npy …`
The default `train_config.yaml` consumes this directly — no config edits needed.

## 3. Audit + quality-filter (tools already in the repo)

```bash
python -m ai.dataset_tools.audit \
    --dataset datasets/sen12ms_cr \
    --output  outputs/metrics/dataset_audit.json
# Read the RECOMMENDATIONS block it prints.

python -m ai.dataset_tools.pair_quality \
    --dataset datasets/sen12ms_cr \
    --output  datasets/sen12ms_cr/pair_quality.json
```

To drop bad pairs during training, uncomment in `ai/restormer/train_config.yaml`:

```yaml
data:
  pair_quality_path: "datasets/sen12ms_cr/pair_quality.json"
  min_pair_quality: 0.5
```

## 4. Smoke test (prove the loss descends — works on CPU)

```bash
python -m ai.restormer.train --smoke --config ai/restormer/train_config.yaml
```

`--smoke` = tiny subset (64), 2 epochs, Phase 1 only, AMP off. It must finish,
report a mean loss, validate, and write `phase1_final.pt`. If it NaNs or errors,
fix the data before committing to a full run. (`--max-samples N` caps any run.)

## 5. Full training (GPU)

```bash
python -m ai.restormer.train --config ai/restormer/train_config.yaml
# In another terminal:
tensorboard --logdir outputs/metrics
```

- Phase 1 (warmup, 20 ep) → Phase 2 (main, 150 ep) → Phase 3 (LISS-IV fine-tune,
  runs only if `datasets/liss4/` exists, else skipped).
- **Watch:** validation **cloud-region PSNR**. Target ≥28 dB by ~epoch 50.
- Gradient clipping, EMA (saved in checkpoints + applied at validation), MS-SSIM
  NaN guards, and per-phase checkpointing are already wired in — nothing to add.

## 6. Serve the trained model

Training writes `ai/restormer/checkpoints/phase2_final.pt` — **exactly** the path
`docker-compose.yml` and `backend/config/settings.py` expect, so no rename.

```bash
export RESTORMER_CHECKPOINT=ai/restormer/checkpoints/phase2_final.pt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The demo pipeline (`detect → reconstruct → metrics → report`) now uses the real
model instead of the clear-region-mean fallback.

## 7. Teammate handoff: LISS-4 fine-tuning

The trained base checkpoint in this workspace is:

- `ai/restormer/checkpoints/phase2_final.pt`
- Size: `407741217` bytes
- SHA256: `1B377C3D43D6F0943CEA5D1D07FF907DFF165CFEBBDE14E6EA224C4DD849AA74`

Since checkpoints are excluded from git, hand off this file out-of-band and
verify with the SHA256 above.

Once LISS-4 data is prepared in `datasets/liss4/`, continue training with:

```bash
python -m ai.restormer.train \
  --config ai/restormer/train_config.yaml \
  --resume ai/restormer/checkpoints/phase2_final.pt
```

This continues from the trained base model and enables Phase 3 when LISS-4 is
present.

---

### Tuning knobs (`ai/restormer/train_config.yaml`)

| Symptom | Knob |
|---|---|
| GPU OOM | `phase*.batch_size` 8→4→2, or `data.patch_size` 256→128 |
| Loss NaN at epoch 1 | already guarded; if it persists, lower `phase1.lr` |
| Cloud PSNR stagnates while global rises | raise `loss.w_sam` 0.5→0.8 |
| Slow convergence | confirm `pair_quality` filtering is on (step 3) |
