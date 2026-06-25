"""
Scientific verdict tiers for cloud-removal results.

Standard practice in remote-sensing evaluation is to report raw metrics and let
a reviewer judge them. For the ISRO demo we additionally distil the metric suite
into a single, defensible verdict so a non-specialist can read the result at a
glance — without ever hiding the underlying numbers.

Verdict tiers
-------------
    "Excellent"                — meets state-of-the-art SEN2-CR targets
    "Scientifically Acceptable"— usable for analysis with documented caveats
    "Needs Improvement"        — below the bar for quantitative use

The verdict is driven primarily by **cloud-region** metrics (the actual
reconstruction task), falling back to global metrics when no cloud mask was
supplied. Reference target ranges follow the project evaluation protocol
(docs/evaluation_protocol.md): PSNR 28-34 dB, SSIM 0.88-0.94, SAM 3-7 deg.

The thresholds here are intentionally conservative — for a hackathon it is far
better to under-claim ("Scientifically Acceptable") than to over-claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# Verdict tier labels (exported for reuse by the frontend / report builder).
EXCELLENT = "Excellent"
ACCEPTABLE = "Scientifically Acceptable"
NEEDS_IMPROVEMENT = "Needs Improvement"

# Tier → numeric score used for aggregation.
_TIER_SCORE = {EXCELLENT: 2.0, ACCEPTABLE: 1.0, NEEDS_IMPROVEMENT: 0.0}


@dataclass(frozen=True)
class Criterion:
    """A single metric evaluated against excellent / acceptable thresholds."""

    key: str                 # metric dict key
    label: str               # human-readable name
    unit: str                # display unit
    excellent: float         # threshold for the Excellent tier
    acceptable: float        # threshold for the Acceptable tier
    higher_is_better: bool   # direction of "better"
    weight: float = 1.0      # contribution to the overall verdict

    def tier(self, value: float) -> str:
        if self.higher_is_better:
            if value >= self.excellent:
                return EXCELLENT
            if value >= self.acceptable:
                return ACCEPTABLE
            return NEEDS_IMPROVEMENT
        # lower is better
        if value <= self.excellent:
            return EXCELLENT
        if value <= self.acceptable:
            return ACCEPTABLE
        return NEEDS_IMPROVEMENT


# Primary criteria: cloud-region reconstruction quality (the real task).
_CLOUD_CRITERIA = [
    Criterion("cloud_psnr_db", "Cloud-region PSNR", "dB", 30.0, 26.0, True, weight=1.0),
    Criterion("cloud_ssim", "Cloud-region SSIM", "", 0.90, 0.80, True, weight=1.0),
    Criterion("cloud_sam_deg", "Cloud-region SAM", "deg", 5.0, 8.0, False, weight=1.0),
]

# Fallback criteria when no cloud mask was provided (global image quality).
_GLOBAL_CRITERIA = [
    Criterion("psnr_db", "Global PSNR", "dB", 30.0, 26.0, True, weight=1.0),
    Criterion("ssim", "Global SSIM", "", 0.90, 0.80, True, weight=1.0),
    Criterion("sam_deg", "Global SAM", "deg", 5.0, 8.0, False, weight=1.0),
]

# Secondary criteria: supporting evidence (lower weight, never the sole driver).
_SECONDARY_CRITERIA = [
    Criterion("clear_psnr_db", "Clear-region preservation", "dB", 40.0, 34.0, True, weight=0.5),
    Criterion("veg_ndvi_mae", "Vegetation NDVI MAE", "", 0.03, 0.06, False, weight=0.5),
    Criterion("improvement_db", "Gain vs cloudy baseline", "dB", 3.0, 0.5, True, weight=0.5),
]


@dataclass
class CriterionAssessment:
    """Per-metric assessment result (serialisable)."""

    key: str
    label: str
    value: float
    unit: str
    tier: str
    target: str

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "value": round(self.value, 4),
            "unit": self.unit,
            "tier": self.tier,
            "target": self.target,
        }


@dataclass
class ScientificVerdict:
    """Overall verdict plus the per-criterion breakdown that produced it."""

    verdict: str
    score: float                         # normalised 0..1
    headline: str
    assessments: list[CriterionAssessment] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    primary_basis: str = "cloud_region"  # 'cloud_region' or 'global'

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "score": round(self.score, 4),
            "headline": self.headline,
            "primary_basis": self.primary_basis,
            "criteria": [a.to_dict() for a in self.assessments],
            "notes": self.notes,
        }


def _is_valid(value: Optional[float]) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _format_target(c: Criterion) -> str:
    arrow = ">=" if c.higher_is_better else "<="
    unit = f" {c.unit}" if c.unit else ""
    return f"{arrow} {c.excellent:g}{unit} (Excellent), {arrow} {c.acceptable:g}{unit} (Acceptable)"


def summarize(metrics: dict[str, float]) -> ScientificVerdict:
    """
    Produce a :class:`ScientificVerdict` from a metrics dict.

    Args:
        metrics: Output of ``ai.validation.cloud_region.full_scientific_metrics``
                 (or any superset containing the relevant keys).

    Returns:
        ScientificVerdict with overall tier, normalised score, per-criterion
        breakdown, and human-readable notes.
    """
    # Choose the primary basis: cloud-region metrics if present & valid.
    cloud_available = any(_is_valid(metrics.get(c.key)) for c in _CLOUD_CRITERIA)
    primary = _CLOUD_CRITERIA if cloud_available else _GLOBAL_CRITERIA
    basis = "cloud_region" if cloud_available else "global"

    assessments: list[CriterionAssessment] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for c in primary + _SECONDARY_CRITERIA:
        value = metrics.get(c.key)
        if not _is_valid(value):
            continue
        value = float(value)
        tier = c.tier(value)
        assessments.append(
            CriterionAssessment(
                key=c.key,
                label=c.label,
                value=value,
                unit=c.unit,
                tier=tier,
                target=_format_target(c),
            )
        )
        weighted_sum += _TIER_SCORE[tier] * c.weight
        weight_total += c.weight

    if weight_total == 0.0:
        return ScientificVerdict(
            verdict=NEEDS_IMPROVEMENT,
            score=0.0,
            headline="Insufficient metrics to assess reconstruction quality.",
            assessments=[],
            notes=["No valid metrics were available (no ground-truth reference?)."],
            primary_basis=basis,
        )

    # Normalise to 0..1 (2.0 == every criterion Excellent).
    score = weighted_sum / (weight_total * 2.0)

    if score >= 0.85:
        verdict = EXCELLENT
    elif score >= 0.5:
        verdict = ACCEPTABLE
    else:
        verdict = NEEDS_IMPROVEMENT

    notes = _build_notes(metrics, assessments, basis)
    headline = _build_headline(verdict, basis, assessments)

    return ScientificVerdict(
        verdict=verdict,
        score=score,
        headline=headline,
        assessments=assessments,
        notes=notes,
        primary_basis=basis,
    )


def _build_headline(verdict: str, basis: str, assessments: list[CriterionAssessment]) -> str:
    basis_txt = "cloud-region" if basis == "cloud_region" else "global"
    if verdict == EXCELLENT:
        return f"Reconstruction meets state-of-the-art targets on {basis_txt} metrics."
    if verdict == ACCEPTABLE:
        return f"Reconstruction is usable for analysis ({basis_txt} metrics within acceptable range)."
    return f"Reconstruction is below the quantitative-use threshold on {basis_txt} metrics."


def _build_notes(
    metrics: dict[str, float],
    assessments: list[CriterionAssessment],
    basis: str,
) -> list[str]:
    notes: list[str] = []

    if basis == "global":
        notes.append(
            "No cloud mask supplied — verdict is based on global image quality, "
            "which can be optimistic since clear pixels dominate the scene."
        )

    # Flag any criterion that fell to the lowest tier.
    weak = [a.label for a in assessments if a.tier == NEEDS_IMPROVEMENT]
    if weak:
        notes.append("Below target: " + ", ".join(weak) + ".")

    # Clear-region preservation is a quality gate of its own.
    clear_psnr = metrics.get("clear_psnr_db")
    if _is_valid(clear_psnr) and float(clear_psnr) < 34.0:
        notes.append(
            f"Clear-region PSNR is {float(clear_psnr):.1f} dB — the model is altering "
            "cloud-free pixels more than it should (target >= 40 dB)."
        )

    # NDVI bias direction is scientifically meaningful for agriculture.
    bias = metrics.get("veg_ndvi_bias")
    if _is_valid(bias) and abs(float(bias)) > 0.02:
        direction = "over-estimated" if float(bias) > 0 else "under-estimated"
        notes.append(
            f"Vegetation NDVI is {direction} by {abs(float(bias)):.03f} on average."
        )

    if not notes:
        notes.append("All evaluated criteria are within target ranges.")
    return notes
