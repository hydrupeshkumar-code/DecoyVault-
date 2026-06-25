"""
Synthetic LISS-IV demo scene generator.

Produces three small, deterministic demo scenes so the full pipeline
(upload -> detect -> reconstruct -> metrics -> report) can be demonstrated
without any real data, model checkpoint, or GPU.

Scenes (band order is ALWAYS Green=0, Red=1, NIR=2 — LISS-IV convention):
    agriculture  — high-NDVI field mosaic
    urban        — low-NDVI built-up grid
    water        — near-zero / negative NDVI smooth water body

For each scene this writes:
    datasets/demo/<scene>/clear.npy   [3, H, W] float32 in [0,1]  (ground truth)
    datasets/demo/<scene>/cloudy.npy  [3, H, W] float32 in [0,1]  (clouds added)
    datasets/demo/<scene>/mask.npy    [H, W]    float32 binary    (1 = cloud)
    frontend/public/demo/<scene>_before.png   false-colour cloudy preview
    frontend/public/demo/<scene>_after.png    false-colour clear preview

The backend resolves a file ID of ``demo_<scene>`` to these arrays
(see backend/services/file_resolver.py).

Usage:
    python -m datasets.demo.generate_demo
    python datasets/demo/generate_demo.py --size 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:  # thumbnails are best-effort
    _HAS_PIL = False

SCENES = ("agriculture", "urban", "water")
SEED = 42

# Repo paths (this file lives at datasets/demo/generate_demo.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEMO_ROOT = _REPO_ROOT / "datasets" / "demo"
_THUMB_ROOT = _REPO_ROOT / "frontend" / "public" / "demo"


# ---------------------------------------------------------------------------
# Procedural scene synthesis  (returns [3, H, W] float32 in [0,1])
# ---------------------------------------------------------------------------

def _coords(size: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.linspace(0, 1, size, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, size, dtype=np.float32)[None, :]
    return y, x


def _clip01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _agriculture(size: int, rng: np.random.Generator) -> np.ndarray:
    """High-NDVI farmland: strong NIR, low Red, blocky fields."""
    y, x = _coords(size)
    # Blocky field pattern via quantised low-frequency sinusoids.
    fields = (np.sin(x * 18) * np.cos(y * 14) + np.sin((x + y) * 9))
    fields = (np.round(fields * 1.5) / 1.5)  # quantise into "plots"
    fields = (fields - fields.min()) / (np.ptp(fields) + 1e-6)

    green = 0.18 + 0.10 * fields + 0.02 * rng.standard_normal((size, size))
    red   = 0.10 + 0.08 * fields + 0.02 * rng.standard_normal((size, size))
    nir   = 0.45 + 0.25 * fields + 0.03 * rng.standard_normal((size, size))  # vegetation bright in NIR
    return _clip01(np.stack([green, red, nir], axis=0))


def _urban(size: int, rng: np.random.Generator) -> np.ndarray:
    """Low-NDVI built-up area: grid of blocks, moderate all bands."""
    y, x = _coords(size)
    grid = ((np.sin(x * 40) > 0.2).astype(np.float32) * (np.sin(y * 40) > 0.2).astype(np.float32))
    roads = ((np.sin(x * 40) < -0.9) | (np.sin(y * 40) < -0.9)).astype(np.float32)
    base = 0.22 + 0.18 * grid - 0.05 * roads

    green = base + 0.02 * rng.standard_normal((size, size))
    red   = base + 0.03 + 0.02 * rng.standard_normal((size, size))  # built-up: red >= green
    nir   = base + 0.06 + 0.03 * rng.standard_normal((size, size))  # only slightly higher NIR
    return _clip01(np.stack([green, red, nir], axis=0))


def _water(size: int, rng: np.random.Generator) -> np.ndarray:
    """Negative/near-zero NDVI water body: dark, smooth, low NIR.

    The scene is predominantly water (low NIR → negative NDVI) with only a thin
    vegetated land strip along the top edge.
    """
    y, x = _coords(size)
    ripples = 0.015 * np.sin(x * 60 + y * 20)
    land = np.clip((0.16 - y) * 8, 0, 1)  # vegetated land in the top ~16% only

    green = 0.06 + 0.06 * land + ripples + 0.005 * rng.standard_normal((size, size))
    red   = 0.05 + 0.04 * land + ripples + 0.005 * rng.standard_normal((size, size))
    # Water strongly absorbs NIR (NDVI < 0); the land strip is bright in NIR.
    nir   = 0.03 + 0.45 * land + 0.005 * rng.standard_normal((size, size))
    return _clip01(np.stack([green, red, nir], axis=0))


_GENERATORS = {"agriculture": _agriculture, "urban": _urban, "water": _water}


# ---------------------------------------------------------------------------
# Cloud synthesis
# ---------------------------------------------------------------------------

def _gaussian_blob(size: int, cy: float, cx: float, sy: float, sx: float) -> np.ndarray:
    y, x = _coords(size)
    return np.exp(-(((y - cy) ** 2) / (2 * sy ** 2) + ((x - cx) ** 2) / (2 * sx ** 2)))


def _make_clouds(size: int, rng: np.random.Generator) -> np.ndarray:
    """Soft cloud opacity field in [0,1] from a few overlapping blobs."""
    opacity = np.zeros((size, size), dtype=np.float32)
    n_blobs = rng.integers(3, 6)
    for _ in range(n_blobs):
        cy, cx = rng.uniform(0.25, 0.75, size=2)
        sy, sx = rng.uniform(0.08, 0.20, size=2)
        opacity += _gaussian_blob(size, cy, cx, sy, sx).astype(np.float32)
    opacity = opacity / (opacity.max() + 1e-6)
    # Sharpen a little so there are clearly-clouded and clearly-clear areas.
    opacity = _clip01((opacity - 0.25) / 0.6)
    return opacity


def _apply_clouds(clear: np.ndarray, opacity: np.ndarray) -> np.ndarray:
    """Blend the clear scene toward bright cloud reflectance by opacity."""
    cloud_reflectance = np.array([0.85, 0.85, 0.88], dtype=np.float32)[:, None, None]
    op = opacity[None, :, :]
    cloudy = clear * (1.0 - op) + cloud_reflectance * op
    return _clip01(cloudy)


# ---------------------------------------------------------------------------
# Thumbnails (false-colour CIR: R<-NIR, G<-Red, B<-Green)
# ---------------------------------------------------------------------------

def _false_colour(chw: np.ndarray) -> np.ndarray:
    """[3,H,W] (G,R,NIR) -> [H,W,3] uint8 CIR composite (veg appears red)."""
    green, red, nir = chw[0], chw[1], chw[2]
    rgb = np.stack([nir, red, green], axis=-1)  # NIR->R, Red->G, Green->B
    # Light per-channel stretch for visibility.
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def _save_thumbnail(chw: np.ndarray, path: Path) -> None:
    if not _HAS_PIL:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_false_colour(chw)).save(path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def generate_scene(scene: str, size: int, rng: np.random.Generator) -> dict[str, float]:
    clear = _GENERATORS[scene](size, rng)
    opacity = _make_clouds(size, rng)
    cloudy = _apply_clouds(clear, opacity)
    mask = (opacity > 0.5).astype(np.float32)

    scene_dir = _DEMO_ROOT / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    np.save(scene_dir / "clear.npy", clear)
    np.save(scene_dir / "cloudy.npy", cloudy)
    np.save(scene_dir / "mask.npy", mask)

    _save_thumbnail(cloudy, _THUMB_ROOT / f"{scene}_before.png")
    _save_thumbnail(clear, _THUMB_ROOT / f"{scene}_after.png")

    nir, red = clear[2], clear[1]
    ndvi = float(((nir - red) / (nir + red + 1e-6)).mean())
    return {"cloud_fraction": float(mask.mean()), "mean_ndvi": ndvi}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic LISS-IV demo scenes")
    parser.add_argument("--size", type=int, default=256, help="Scene height/width in pixels")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Generating {len(SCENES)} demo scenes ({args.size}x{args.size}) ...")
    for scene in SCENES:
        stats = generate_scene(scene, args.size, rng)
        print(
            f"  {scene:12s} cloud_fraction={stats['cloud_fraction']:.2f} "
            f"mean_NDVI={stats['mean_ndvi']:+.2f}"
        )
    thumb_note = "" if _HAS_PIL else "  (Pillow not installed — thumbnails skipped)"
    print(f"Done. Arrays -> {_DEMO_ROOT}  Thumbnails -> {_THUMB_ROOT}{thumb_note}")


if __name__ == "__main__":
    main()
