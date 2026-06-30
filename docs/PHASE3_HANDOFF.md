# Phase 3 Handoff (LISS-4 Fine-Tuning)

This repo already has a trained base checkpoint from Phase 2.

## Final checkpoint to use

- Path: `ai/restormer/checkpoints/phase2_final.pt`
- Size: `407741217` bytes
- SHA256: `1B377C3D43D6F0943CEA5D1D07FF907DFF165CFEBBDE14E6EA224C4DD849AA74`

Note: model checkpoints are not committed to git (large binary).
Share `phase2_final.pt` directly with your teammate.

## Teammate steps for last phase (Phase 3)

1. Put prepared LISS-4 pairs under `datasets/liss4/`.
2. Copy checkpoint to `ai/restormer/checkpoints/phase2_final.pt`.
3. Run:

```bash
python -m ai.restormer.train \
  --config ai/restormer/train_config.yaml \
  --resume ai/restormer/checkpoints/phase2_final.pt
```

Phase 3 starts automatically when LISS-4 data exists at `phase3.liss4_root`
(default `datasets/liss4`).
