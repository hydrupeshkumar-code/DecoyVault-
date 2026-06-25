"""
Dataset health audit for cloud removal training data.

Produces a comprehensive JSON report covering:
  - Total sample count and split statistics
  - Cloud fraction distribution (histogram)
  - Per-band statistics (mean, std, min, max) across cloudy and clear sets
  - Pair quality score distribution
  - Flagged samples (missing files, shape mismatches, NaN/Inf values)
  - NDVI statistics (vegetation cover characterization)
  - Recommended min_pair_quality threshold

Run this BEFORE training to verify dataset integrity.

Usage:
    python -m ai.dataset_tools.audit \
        --dataset datasets/sen12ms_cr \
        --output  outputs/metrics/dataset_audit.json

Or with quality scores pre-computed:
    python -m ai.dataset_tools.audit \
        --dataset datasets/sen12ms_cr \
        --quality datasets/sen12ms_cr/pair_quality.json \
        --output  outputs/metrics/dataset_audit.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class BandStats:
    mean: float = 0.0
    std: float  = 0.0
    p2: float   = 0.0
    p98: float  = 0.0
    min: float  = 0.0
    max: float  = 0.0
    n_nan: int  = 0
    n_inf: int  = 0


@dataclass
class DatasetAuditReport:
    dataset_path: str = ""
    n_total: int = 0
    n_valid_triplets: int = 0
    n_flagged: int = 0
    flagged_samples: list[dict] = field(default_factory=list)

    cloud_fraction_mean: float = 0.0
    cloud_fraction_std: float  = 0.0
    cloud_fraction_p10: float  = 0.0
    cloud_fraction_p90: float  = 0.0

    cloudy_band_stats: list[BandStats] = field(default_factory=lambda: [BandStats(), BandStats(), BandStats()])
    clear_band_stats:  list[BandStats] = field(default_factory=lambda: [BandStats(), BandStats(), BandStats()])

    ndvi_mean_cloudy: float = 0.0
    ndvi_mean_clear:  float = 0.0
    ndvi_diff_mean:   float = 0.0

    pair_quality_mean: float = 1.0
    pair_quality_p25:  float = 1.0
    recommended_min_quality: float = 0.5

    shape_mismatches: list[str] = field(default_factory=list)
    recommendations: list[str]  = field(default_factory=list)


class DatasetAuditor:
    """
    Comprehensive dataset health checker for cloud removal training data.

    Args:
        sample_limit:  Maximum number of samples to inspect (set for speed).
        quality_path:  Path to pre-computed pair_quality.json.
    """

    def __init__(
        self,
        sample_limit: Optional[int] = None,
        quality_path: Optional[str | Path] = None,
    ) -> None:
        self.sample_limit = sample_limit
        self.quality_scores: dict[str, float] = {}
        if quality_path:
            self._load_quality(quality_path)

    # ------------------------------------------------------------------

    def audit_directory(
        self,
        root_dir: str | Path,
        cloudy_subdir: str = "cloudy",
        clear_subdir: str = "clear",
        mask_subdir: str = "masks",
    ) -> DatasetAuditReport:
        """
        Audit a flat-layout dataset directory.

        Returns a DatasetAuditReport with all statistics.
        """
        from ai.geospatial.tiff_io import load_any

        root = Path(root_dir)
        cloudy_dir = root / cloudy_subdir
        clear_dir  = root / clear_subdir
        mask_dir   = root / mask_subdir

        exts = {".tif", ".tiff", ".npy"}
        all_stems = sorted(p.stem for p in cloudy_dir.iterdir() if p.suffix.lower() in exts)
        if self.sample_limit:
            all_stems = all_stems[:self.sample_limit]

        report = DatasetAuditReport(dataset_path=str(root), n_total=len(all_stems))

        cloud_fractions: list[float] = []
        cloudy_bands: list[list[float]] = [[], [], []]
        clear_bands:  list[list[float]] = [[], [], []]
        ndvi_cloudy:  list[float] = []
        ndvi_clear:   list[float] = []
        quality_vals: list[float] = []
        flagged: list[dict] = []
        shape_issues: list[str] = []
        n_valid = 0

        for stem in all_stems:
            issues: list[str] = []

            try:
                c_path = _find_file(cloudy_dir, stem)
                cloudy, _ = load_any(c_path, normalize=True)
            except Exception as e:
                flagged.append({"stem": stem, "issue": f"cloudy_load_error: {e}"})
                continue

            try:
                cl_path = _find_file(clear_dir, stem)
                clear, _ = load_any(cl_path, normalize=True)
            except Exception as e:
                flagged.append({"stem": stem, "issue": f"clear_load_error: {e}"})
                continue

            try:
                m_path = _find_file(mask_dir, stem)
                mask, _ = load_any(m_path, normalize=False)
                mask = mask.squeeze()
                mask = (mask > 0.5).astype(np.float32)
            except Exception as e:
                flagged.append({"stem": stem, "issue": f"mask_load_error: {e}"})
                continue

            # Shape consistency
            if cloudy.shape != clear.shape:
                issues.append(f"shape_mismatch cloudy={cloudy.shape} clear={clear.shape}")
                shape_issues.append(f"{stem}: cloudy={cloudy.shape} vs clear={clear.shape}")
            if mask.shape != cloudy.shape[1:]:
                issues.append(f"mask_shape_mismatch mask={mask.shape} vs image={cloudy.shape[1:]}")

            # NaN/Inf
            for name, arr in [("cloudy", cloudy), ("clear", clear), ("mask", mask)]:
                if np.any(np.isnan(arr)):
                    issues.append(f"NaN in {name}")
                if np.any(np.isinf(arr)):
                    issues.append(f"Inf in {name}")

            if issues:
                flagged.append({"stem": stem, "issues": issues})

            # Accumulate statistics (regardless of minor issues)
            cf = float(mask.mean())
            cloud_fractions.append(cf)

            for b in range(min(3, cloudy.shape[0])):
                cloudy_bands[b].extend(cloudy[b].flatten().tolist()[::16])  # subsample
                clear_bands[b].extend(clear[b].flatten().tolist()[::16])

            if cloudy.shape[0] >= 3:
                ndvi_cloudy.append(_ndvi(cloudy))
                ndvi_clear.append(_ndvi(clear))

            if stem in self.quality_scores:
                quality_vals.append(self.quality_scores[stem])

            n_valid += 1

        report.n_valid_triplets = n_valid
        report.n_flagged = len(flagged)
        report.flagged_samples = flagged[:50]  # truncate for readability

        if cloud_fractions:
            cfa = np.array(cloud_fractions)
            report.cloud_fraction_mean = float(cfa.mean())
            report.cloud_fraction_std  = float(cfa.std())
            report.cloud_fraction_p10  = float(np.percentile(cfa, 10))
            report.cloud_fraction_p90  = float(np.percentile(cfa, 90))

        report.cloudy_band_stats = [_band_stats(cloudy_bands[b]) for b in range(3)]
        report.clear_band_stats  = [_band_stats(clear_bands[b])  for b in range(3)]

        if ndvi_cloudy:
            report.ndvi_mean_cloudy = float(np.mean(ndvi_cloudy))
            report.ndvi_mean_clear  = float(np.mean(ndvi_clear))
            report.ndvi_diff_mean   = float(np.mean(np.array(ndvi_clear) - np.array(ndvi_cloudy)))

        if quality_vals:
            qa = np.array(quality_vals)
            report.pair_quality_mean = float(qa.mean())
            report.pair_quality_p25  = float(np.percentile(qa, 25))
            report.recommended_min_quality = float(np.percentile(qa, 20))

        report.shape_mismatches = shape_issues
        report.recommendations = self._generate_recommendations(report)

        log.info(
            "Audit complete: %d total, %d valid, %d flagged",
            report.n_total,
            report.n_valid_triplets,
            report.n_flagged,
        )
        return report

    @staticmethod
    def save_report(report: DatasetAuditReport, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _serialize(obj):
            if isinstance(obj, BandStats):
                return asdict(obj)
            raise TypeError(f"Object {obj!r} is not JSON serializable")

        with open(output_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=_serialize)
        log.info("Audit report saved → %s", output_path)

    # ------------------------------------------------------------------

    def _load_quality(self, path: str | Path) -> None:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "scores" in data:
            self.quality_scores = {s: v["score"] for s, v in data["scores"].items()}
        elif isinstance(data, dict):
            self.quality_scores = data

    @staticmethod
    def _generate_recommendations(report: DatasetAuditReport) -> list[str]:
        recs: list[str] = []
        if report.n_flagged > report.n_total * 0.05:
            recs.append(
                f"HIGH: {report.n_flagged}/{report.n_total} samples flagged. "
                "Run pair_quality.py and filter before training."
            )
        if report.cloud_fraction_mean < 0.10:
            recs.append(
                f"WARNING: Mean cloud fraction is {report.cloud_fraction_mean:.2f}. "
                "Dataset may lack sufficient cloud coverage for effective training. "
                "Apply min_cloud_fraction=0.05 filter."
            )
        if report.cloud_fraction_mean > 0.85:
            recs.append(
                f"WARNING: Mean cloud fraction is {report.cloud_fraction_mean:.2f}. "
                "Dataset has near-total cloud cover — limited clear reference signal. "
                "Apply max_cloud_fraction=0.90 filter."
            )
        if report.ndvi_diff_mean > 0.1:
            recs.append(
                f"WARNING: Mean NDVI difference cloudy→clear is {report.ndvi_diff_mean:.3f}. "
                "Possible temporal mismatch from seasonal vegetation changes."
            )
        if report.pair_quality_p25 < 0.5:
            recs.append(
                f"WARNING: 25th percentile pair quality is {report.pair_quality_p25:.3f}. "
                f"Recommend setting min_pair_quality >= {report.recommended_min_quality:.2f}."
            )
        if report.shape_mismatches:
            recs.append(
                f"ERROR: {len(report.shape_mismatches)} shape mismatches between cloudy/clear. "
                "These pairs will cause training failures. Remove or re-process."
            )
        if not recs:
            recs.append("Dataset appears healthy. No critical issues found.")
        return recs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ndvi(arr: np.ndarray) -> float:
    """NDVI from [C, H, W] array with bands [Green, Red, NIR]."""
    red = arr[1].mean()
    nir = arr[2].mean()
    denom = nir + red
    return float((nir - red) / (denom + 1e-8)) if denom > 1e-8 else 0.0


def _band_stats(values: list[float]) -> BandStats:
    if not values:
        return BandStats()
    arr = np.array(values, dtype=np.float32)
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return BandStats(n_nan=int(np.sum(np.isnan(arr))), n_inf=int(np.sum(np.isinf(arr))))
    return BandStats(
        mean=float(valid.mean()),
        std=float(valid.std()),
        p2=float(np.percentile(valid, 2)),
        p98=float(np.percentile(valid, 98)),
        min=float(valid.min()),
        max=float(valid.max()),
        n_nan=int(np.sum(np.isnan(arr))),
        n_inf=int(np.sum(np.isinf(arr))),
    )


def _find_file(directory: Path, stem: str) -> Path:
    for ext in (".tif", ".tiff", ".npy"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No file with stem '{stem}' in {directory}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Audit cloud removal training dataset")
    parser.add_argument("--dataset",  required=True)
    parser.add_argument("--cloudy",   default="cloudy")
    parser.add_argument("--clear",    default="clear")
    parser.add_argument("--masks",    default="masks")
    parser.add_argument("--quality",  default=None, help="Path to pair_quality.json")
    parser.add_argument("--output",   default="outputs/metrics/dataset_audit.json")
    parser.add_argument("--max",      type=int, default=None)
    args = parser.parse_args()

    auditor = DatasetAuditor(sample_limit=args.max, quality_path=args.quality)
    report = auditor.audit_directory(
        root_dir=args.dataset,
        cloudy_subdir=args.cloudy,
        clear_subdir=args.clear,
        mask_subdir=args.masks,
    )
    auditor.save_report(report, args.output)

    print("\n=== DATASET AUDIT SUMMARY ===")
    print(f"  Total samples:    {report.n_total}")
    print(f"  Valid triplets:   {report.n_valid_triplets}")
    print(f"  Flagged:          {report.n_flagged}")
    print(f"  Cloud fraction:   {report.cloud_fraction_mean:.3f} ± {report.cloud_fraction_std:.3f}")
    print(f"  NDVI (clear):     {report.ndvi_mean_clear:.3f}")
    print("\n  RECOMMENDATIONS:")
    for r in report.recommendations:
        print(f"    • {r}")


if __name__ == "__main__":
    main()
