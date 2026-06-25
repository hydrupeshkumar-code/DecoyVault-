"""
Dataset pair quality scoring for cloud removal training data.

BACKGROUND
----------
The primary cause of performance degradation in cloud removal models is NOT
model architecture — it is cloudy/clear pair mismatch. Common sources:

  1. Temporal mismatch: cloudy and clear acquisitions are from different dates.
     Seasonal changes (vegetation phenology, snow cover) cause ground-truth
     inconsistency that is impossible for the model to learn.

  2. Registration error: sub-pixel or multi-pixel spatial misalignment between
     the cloudy and clear tiles, which destroys per-pixel supervision.

  3. Remaining cloud in the "clear" reference: partially cloudy references
     introduce false supervision — the model is asked to hallucinate a clear
     sky that the reference does not actually show.

  4. Radiometric inconsistency: atmospheric correction artifacts, sensor
     gain differences, or BRDF effects creating spectral mismatches between
     the pair that are unrelated to clouds.

This module scores each (cloudy, clear, mask) triplet on [0, 1] where:
    1.0 = perfect pair, safe to use for training
    0.5 = borderline pair, use with caution
    0.0 = corrupted pair, discard

USAGE
-----
    from ai.dataset_tools.pair_quality import PairQualityScorer
    scorer = PairQualityScorer()
    scores = scorer.score_directory("datasets/sen12ms_cr")
    scorer.save_scores(scores, "datasets/sen12ms_cr/pair_quality.json")

Then pass `pair_quality.json` to any CloudRemovalDataset via
    dataset.load_quality_scores("datasets/sen12ms_cr/pair_quality.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PairScore:
    """Detailed breakdown of a single pair's quality score."""
    stem: str
    total: float = 1.0
    temporal_consistency: float = 1.0
    registration_quality: float = 1.0
    clear_ref_quality: float = 1.0
    spectral_consistency: float = 1.0
    cloud_fraction: float = 0.0
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "stem": self.stem,
            "score": round(self.total, 4),
            "temporal_consistency": round(self.temporal_consistency, 4),
            "registration_quality": round(self.registration_quality, 4),
            "clear_ref_quality": round(self.clear_ref_quality, 4),
            "spectral_consistency": round(self.spectral_consistency, 4),
            "cloud_fraction": round(self.cloud_fraction, 4),
            "flags": self.flags,
        }


class PairQualityScorer:
    """
    Scores cloudy/clear/mask triplets for training quality.

    Scoring criteria:
        1. Clear-region spectral consistency   (30% weight)
           — Measure spectral distance between cloudy and clear pixels in
             the unmasked (cloud-free) regions. Large spectral differences
             indicate temporal change, not cloud removal quality.

        2. Clear reference cloud contamination (25% weight)
           — If the "clear" reference has mean brightness > threshold in
             masked regions, it likely contains residual cloud or haze.

        3. Spatial registration               (25% weight)
           — Normalized cross-correlation on cloud-free regions.
             Low NCC indicates spatial misalignment.

        4. Cloud fraction sanity              (20% weight)
           — Penalize near-zero cloud fraction (not useful for training)
             and near-total cloud cover (no reference signal remaining).

    Args:
        ncc_threshold:        Minimum NCC for acceptable registration (default 0.85).
        spectral_threshold:   Maximum mean SAM (rad) in clear regions (default 0.15).
        cloud_min:            Minimum useful cloud fraction (default 0.05).
        cloud_max:            Maximum useful cloud fraction (default 0.95).
        residual_cloud_max:   Max mean clear-ref value in masked region (default 0.85).
    """

    def __init__(
        self,
        ncc_threshold: float = 0.85,
        spectral_threshold: float = 0.15,
        cloud_min: float = 0.05,
        cloud_max: float = 0.95,
        residual_cloud_max: float = 0.85,
    ) -> None:
        self.ncc_threshold = ncc_threshold
        self.spectral_threshold = spectral_threshold
        self.cloud_min = cloud_min
        self.cloud_max = cloud_max
        self.residual_cloud_max = residual_cloud_max

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_pair(
        self,
        stem: str,
        cloudy: np.ndarray,
        clear: np.ndarray,
        mask: np.ndarray,
    ) -> PairScore:
        """
        Score a single (cloudy, clear, mask) triplet.

        Args:
            stem:   Sample identifier (for reporting).
            cloudy: [C, H, W] float32 [0,1] cloudy image.
            clear:  [C, H, W] float32 [0,1] clear reference.
            mask:   [1, H, W] or [H, W] float32 binary cloud mask.

        Returns:
            PairScore with per-criterion breakdown.
        """
        score = PairScore(stem=stem)

        # Normalize mask to [H, W]
        m = mask.squeeze() if mask.ndim == 3 else mask
        cloud_fraction = float(m.mean())
        score.cloud_fraction = cloud_fraction
        clear_mask = 1.0 - m  # 1 = cloud-free pixels

        # ── 1. Cloud fraction sanity ──────────────────────────────────
        cf_score = self._cloud_fraction_score(cloud_fraction)
        score.cloud_fraction = cloud_fraction
        if cf_score < 0.3:
            score.flags.append(
                f"cloud_fraction={cloud_fraction:.2f} outside [{self.cloud_min},{self.cloud_max}]"
            )

        # ── 2. Spectral consistency in cloud-free regions ─────────────
        spec_score, spec_val = self._spectral_consistency(cloudy, clear, clear_mask)
        score.temporal_consistency = spec_score
        if spec_score < 0.5:
            score.flags.append(f"high_spectral_change in clear regions: SAM={spec_val:.3f} rad")

        # ── 3. Clear reference quality ────────────────────────────────
        ref_score = self._clear_ref_quality(clear, m)
        score.clear_ref_quality = ref_score
        if ref_score < 0.5:
            score.flags.append("residual_cloud_or_haze in clear reference")

        # ── 4. Registration quality (NCC) ─────────────────────────────
        reg_score, ncc_val = self._registration_quality(cloudy, clear, clear_mask)
        score.registration_quality = reg_score
        if reg_score < 0.5:
            score.flags.append(f"low_registration_NCC={ncc_val:.3f}")

        # ── 5. Spectral consistency (separate from temporal) ──────────
        score.spectral_consistency = spec_score  # reuse

        # ── Aggregate ─────────────────────────────────────────────────
        score.total = float(
            0.30 * spec_score
            + 0.25 * ref_score
            + 0.25 * reg_score
            + 0.20 * cf_score
        )
        return score

    def score_directory(
        self,
        root_dir: str | Path,
        cloudy_subdir: str = "cloudy",
        clear_subdir: str = "clear",
        mask_subdir: str = "masks",
        max_samples: Optional[int] = None,
    ) -> list[PairScore]:
        """
        Score all triplets in a flat-layout dataset directory.

        Args:
            root_dir:      Dataset root (contains cloudy/, clear/, masks/).
            cloudy_subdir: Sub-directory name for cloudy images.
            clear_subdir:  Sub-directory name for clear images.
            mask_subdir:   Sub-directory name for masks.
            max_samples:   Limit to first N samples (for quick audits).

        Returns:
            List of PairScore, one per sample.
        """
        from ai.geospatial.tiff_io import load_any

        root = Path(root_dir)
        cloudy_dir = root / cloudy_subdir
        clear_dir  = root / clear_subdir
        mask_dir   = root / mask_subdir

        exts = {".tif", ".tiff", ".npy"}
        stems = sorted(
            p.stem for p in cloudy_dir.iterdir()
            if p.suffix.lower() in exts
        )
        if max_samples:
            stems = stems[:max_samples]

        scores: list[PairScore] = []
        n = len(stems)

        for i, stem in enumerate(stems):
            try:
                c_path = _find_file(cloudy_dir, stem)
                cl_path = _find_file(clear_dir, stem)
                m_path  = _find_file(mask_dir, stem)

                cloudy, _ = load_any(c_path, normalize=True)
                clear, _  = load_any(cl_path, normalize=True)
                mask, _   = load_any(m_path, normalize=False)

                ps = self.score_pair(stem, cloudy, clear, mask)
                scores.append(ps)

                if (i + 1) % 100 == 0:
                    log.info("Scored %d/%d pairs", i + 1, n)

            except Exception as e:
                log.warning("Failed to score '%s': %s", stem, e)
                scores.append(PairScore(stem=stem, total=0.0, flags=[f"load_error: {e}"]))

        log.info(
            "Scoring complete: %d pairs. Mean score=%.3f, n_rejected=%d (score<0.5)",
            len(scores),
            sum(s.total for s in scores) / max(len(scores), 1),
            sum(1 for s in scores if s.total < 0.5),
        )
        return scores

    @staticmethod
    def save_scores(
        scores: list[PairScore],
        output_path: str | Path,
        format: str = "full",
    ) -> None:
        """
        Save scores to JSON.

        Args:
            scores:       List of PairScore.
            output_path:  Output .json path.
            format:       'full' (detailed) or 'simple' (stem→score only).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "simple":
            data = {s.stem: s.total for s in scores}
        else:
            data = {
                "summary": {
                    "n_total": len(scores),
                    "mean_score": round(sum(s.total for s in scores) / max(len(scores), 1), 4),
                    "n_rejected": sum(1 for s in scores if s.total < 0.5),
                    "n_borderline": sum(1 for s in scores if 0.5 <= s.total < 0.7),
                    "n_good": sum(1 for s in scores if s.total >= 0.7),
                },
                "scores": {s.stem: s.to_dict() for s in scores},
            }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        log.info("Scores saved → %s", output_path)

    # ------------------------------------------------------------------
    # Scoring sub-criteria
    # ------------------------------------------------------------------

    def _cloud_fraction_score(self, cf: float) -> float:
        """Score based on cloud fraction being within useful range."""
        if cf < self.cloud_min:
            return max(0.0, cf / self.cloud_min)
        if cf > self.cloud_max:
            return max(0.0, 1.0 - (cf - self.cloud_max) / (1.0 - self.cloud_max))
        return 1.0

    def _spectral_consistency(
        self,
        cloudy: np.ndarray,
        clear: np.ndarray,
        clear_mask: np.ndarray,
    ) -> tuple[float, float]:
        """
        Measure spectral angle in cloud-free regions.

        Cloud-free pixels should look similar in both images.
        High SAM → temporal change or radiometric inconsistency.
        """
        eps = 1e-8
        # Apply clear mask: only use cloud-free pixels
        valid = clear_mask > 0.5

        if valid.sum() < 100:
            return 0.5, 0.0  # not enough clear pixels to assess

        c_vals = cloudy[:, valid]  # [C, N]
        cl_vals = clear[:, valid]

        dot = (c_vals * cl_vals).sum(axis=0)
        norm_c  = np.linalg.norm(c_vals, axis=0).clip(min=eps)
        norm_cl = np.linalg.norm(cl_vals, axis=0).clip(min=eps)
        cos_angle = np.clip(dot / (norm_c * norm_cl), -1 + eps, 1 - eps)
        sam_mean = float(np.arccos(cos_angle).mean())

        # SAM=0 → perfect, SAM=pi/2 → orthogonal (worst case for reflectance)
        # Threshold at 0.15 rad (~8.6°) as acceptable
        score = max(0.0, 1.0 - sam_mean / self.spectral_threshold)
        return min(1.0, score), sam_mean

    def _clear_ref_quality(
        self,
        clear: np.ndarray,
        cloud_mask: np.ndarray,
    ) -> float:
        """
        Assess whether the 'clear' reference is actually cloud-free.

        If the mean NIR reflectance in masked regions is very high,
        the reference likely has residual cloud/haze.
        """
        valid = cloud_mask > 0.5
        if valid.sum() < 50:
            return 1.0  # no cloud pixels to assess

        # Use NIR band (index 2) — clouds are bright in NIR
        nir_band = clear[2] if clear.shape[0] >= 3 else clear[0]
        masked_nir = nir_band[valid]
        mean_nir = float(masked_nir.mean())

        if mean_nir > self.residual_cloud_max:
            return max(0.0, 1.0 - (mean_nir - self.residual_cloud_max) / (1.0 - self.residual_cloud_max))
        return 1.0

    def _registration_quality(
        self,
        cloudy: np.ndarray,
        clear: np.ndarray,
        clear_mask: np.ndarray,
    ) -> tuple[float, float]:
        """
        Estimate spatial registration via normalized cross-correlation.

        A low NCC in cloud-free regions indicates the tile was cropped from
        a different part of the scene (pair mismatch) or there is sub-pixel
        misregistration.
        """
        valid = clear_mask > 0.5
        if valid.sum() < 200:
            return 0.7, 0.0  # not enough pixels for reliable NCC

        # Use the NIR band (most contrast in vegetation/water boundaries)
        c_band = cloudy[2] if cloudy.shape[0] >= 3 else cloudy[0]
        cl_band = clear[2] if clear.shape[0] >= 3 else clear[0]

        c_vals  = c_band[valid].astype(np.float64)
        cl_vals = cl_band[valid].astype(np.float64)

        # Normalize
        eps = 1e-8
        c_norm  = c_vals  - c_vals.mean()
        cl_norm = cl_vals - cl_vals.mean()
        denom = np.sqrt((c_norm ** 2).sum() * (cl_norm ** 2).sum())

        if denom < eps:
            return 0.5, 0.0

        ncc = float((c_norm * cl_norm).sum() / denom)
        ncc = np.clip(ncc, -1.0, 1.0)

        score = max(0.0, (ncc - self.ncc_threshold) / (1.0 - self.ncc_threshold))
        return min(1.0, score), ncc


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _find_file(directory: Path, stem: str) -> Path:
    for ext in (".tif", ".tiff", ".npy"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No file with stem '{stem}' in {directory}")


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Score cloud removal training pairs")
    parser.add_argument("--dataset", required=True, help="Path to dataset root")
    parser.add_argument("--cloudy",  default="cloudy")
    parser.add_argument("--clear",   default="clear")
    parser.add_argument("--masks",   default="masks")
    parser.add_argument("--output",  default="pair_quality.json")
    parser.add_argument("--max",     type=int, default=None, help="Limit samples")
    parser.add_argument("--format",  choices=["full", "simple"], default="full")
    args = parser.parse_args()

    scorer = PairQualityScorer()
    scores = scorer.score_directory(
        root_dir=args.dataset,
        cloudy_subdir=args.cloudy,
        clear_subdir=args.clear,
        mask_subdir=args.masks,
        max_samples=args.max,
    )
    scorer.save_scores(scores, args.output, format=args.format)


if __name__ == "__main__":
    main()
