"""
Synthetic flat-layout cloud-removal pairs for the smoke test.

The repo ships no bundled training data, so `--smoke` (TRAINING.md step 4) used
to die with a missing-directory error before it could prove the loss descends.
This writes a handful of tiny procedural cloudy/clear/mask pairs in the flat
layout the loaders expect (cloudy/ clear/ masks/, shared stems), so training
runs end-to-end from a fresh clone with no download.

Band order is ALWAYS Green=0, Red=1, NIR=2 (LISS-IV convention), values in [0,1].
Not for real training — geometry-free noise fields, only enough signal for the
model to learn "remove the bright cloud blob".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _scene(size: int, rng: np.random.Generator) -> np.ndarray:
    """[3,H,W] clear scene: blocky fields, vegetation bright in NIR."""
    y = np.linspace(0, 1, size, np.float32)[:, None]
    x = np.linspace(0, 1, size, np.float32)[None, :]
    f = np.sin(x * rng.uniform(8, 20)) * np.cos(y * rng.uniform(6, 16))
    f = (f - f.min()) / (np.ptp(f) + 1e-6)
    green = 0.15 + 0.10 * f
    red = 0.10 + 0.08 * f
    nir = 0.45 + 0.25 * f
    scene = np.stack([green, red, nir], 0) + 0.02 * rng.standard_normal((3, size, size)).astype(np.float32)
    return np.clip(scene, 0, 1).astype(np.float32)


def _clouds(size: int, rng: np.random.Generator) -> np.ndarray:
    """Soft opacity field in [0,1] from a few Gaussian blobs."""
    y = np.linspace(0, 1, size, np.float32)[:, None]
    x = np.linspace(0, 1, size, np.float32)[None, :]
    op = np.zeros((size, size), np.float32)
    for _ in range(int(rng.integers(2, 5))):
        cy, cx = rng.uniform(0.2, 0.8, 2)
        s = rng.uniform(0.10, 0.22)
        op += np.exp(-(((y - cy) ** 2) + ((x - cx) ** 2)) / (2 * s ** 2)).astype(np.float32)
    op = op / (op.max() + 1e-6)
    return np.clip((op - 0.25) / 0.6, 0, 1).astype(np.float32)


def make_synthetic_flat(out_dir: str | Path, n: int = 24, size: int = 128, seed: int = 0) -> Path:
    """
    Write `n` synthetic pairs to out_dir/{cloudy,clear,masks}/pair_XXXX.npy.

    Returns the dataset root (out_dir). Idempotent enough for a smoke run: it
    overwrites existing pair files.
    """
    out = Path(out_dir)
    for sub in ("cloudy", "clear", "masks"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    cloud_refl = np.array([0.85, 0.85, 0.88], np.float32)[:, None, None]
    for i in range(n):
        clear = _scene(size, rng)
        op = _clouds(size, rng)
        cloudy = np.clip(clear * (1 - op[None]) + cloud_refl * op[None], 0, 1).astype(np.float32)
        mask = (op > 0.5).astype(np.float32)  # [H,W]; loader adds channel dim
        stem = f"pair_{i:04d}"
        np.save(out / "cloudy" / f"{stem}.npy", cloudy)
        np.save(out / "clear" / f"{stem}.npy", clear)
        np.save(out / "masks" / f"{stem}.npy", mask)
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate a synthetic flat cloud-removal dataset")
    ap.add_argument("--out", default="datasets/_smoke_synth")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()
    root = make_synthetic_flat(args.out, args.n, args.size)
    print(f"Wrote {args.n} synthetic pairs -> {root}")
