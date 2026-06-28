#!/usr/bin/env python3
"""Check whether the repo is ready for Phase 3 LISS-4 fine-tuning.

Usage:
    python scripts/check_phase3_ready.py
    python scripts/check_phase3_ready.py --run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_CHECKPOINT = Path("ai/restormer/checkpoints/phase2_final.pt")
DEFAULT_LISS4_ROOT = Path("datasets/liss4")


def _find_npy_count(root: Path) -> int:
    return sum(1 for _ in root.rglob("*.npy")) if root.exists() else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Phase 3 readiness")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Checkpoint to resume from")
    parser.add_argument("--liss4-root", default=str(DEFAULT_LISS4_ROOT), help="Root folder for prepared LISS-4 data")
    parser.add_argument("--run", action="store_true", help="Run training if readiness checks pass")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    checkpoint = (repo_root / args.checkpoint).resolve()
    liss4_root = (repo_root / args.liss4_root).resolve()

    issues: list[str] = []
    if not checkpoint.exists():
        issues.append(f"Missing checkpoint: {checkpoint}")
    if not liss4_root.exists():
        issues.append(f"Missing LISS-4 data directory: {liss4_root}")
    else:
        npy_count = _find_npy_count(liss4_root)
        if npy_count == 0:
            issues.append(f"No prepared .npy files found under {liss4_root}")

    print("Phase 3 readiness check")
    print(f"- Checkpoint: {checkpoint} {'OK' if checkpoint.exists() else 'MISSING'}")
    print(f"- LISS-4 data: {liss4_root} {'OK' if liss4_root.exists() and _find_npy_count(liss4_root) > 0 else 'MISSING/EMPTY'}")

    if issues:
        print("\nReady? NO")
        for issue in issues:
            print(f"- {issue}")
        print("\nNext step:")
        print("1. Place the final checkpoint at ai/restormer/checkpoints/phase2_final.pt")
        print("2. Prepare LISS-4 pairs under datasets/liss4/")
        print("3. Re-run: python scripts/check_phase3_ready.py")
        return 1

    print("\nReady? YES")
    print("Training command:")
    print("python -m ai.restormer.train --config ai/restormer/train_config.yaml --resume ai/restormer/checkpoints/phase2_final.pt")

    if args.run:
        print("\nRunning training...")
        subprocess.run(
            [sys.executable, "-m", "ai.restormer.train", "--config", "ai/restormer/train_config.yaml", "--resume", str(checkpoint)],
            cwd=repo_root,
            check=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
