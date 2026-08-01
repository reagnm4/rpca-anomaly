"""Step 2 (Session 4): reduce S to a per-frame score and TEST the core
assumption before any detection metric is computed.

Reduction:  score(t) = sum over pixels of |S[:, t]|   (total sparse energy).

Hypothesis under test: anomalous frames push more mass into the sparse
component than ordinary pedestrian motion does, so their scores sit HIGHER.
This script builds the pooled score vector, re-proves it is aligned with the
labels, and draws density-normalized split histograms (normal vs anomalous).

REJECT GATE: if the two distributions overlap heavily -- or the direction is
inverted (anomalous mean at or below normal mean) -- the raw-magnitude
reduction is rejected and we pivot to a spatial-concentration measure
(nonzero-pixel count / connected-component size) BEFORE computing ROC-AUC or
any other detection metric. The overlap/effect-size numbers below are
distribution diagnostics, not detection metrics.

First run decomposes all 12 test clips with IALM (minutes) and caches X/L/S to
outputs/; later runs reuse the cache.

    python scripts/score_and_histograms.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from rpca_anomaly.evaluation import (  # noqa: E402
    assert_alignment,
    build_labels,
    build_scores,
    split_by_label,
)

FIG = Path("figures/ped2_energy_hist.png")
NORMAL_COLOR = "#4C78A8"
ANOM_COLOR = "#E45756"


def describe(name: str, x: np.ndarray) -> str:
    return (
        f"{name:<10} n={x.size:>5}  mean={x.mean():.5g}  median={np.median(x):.5g}  "
        f"std={x.std():.5g}  min={x.min():.5g}  max={x.max():.5g}"
    )


def main() -> None:
    labels, index_map = build_labels()
    scores = build_scores(verbose=True)  # score(t) = sum|S|, cached per clip
    print()
    # Re-prove alignment now that real scores exist (Step-1 was length-only).
    assert_alignment(labels, scores, index_map)

    normal, anom = split_by_label(scores, labels)

    print("\nper-class score distribution  (score = sum|S| per frame):")
    print(" ", describe("normal", normal))
    print(" ", describe("anomalous", anom))

    # --- Distribution-overlap diagnostics (NOT detection metrics) ---
    pooled_var = (normal.var() * normal.size + anom.var() * anom.size) / (
        normal.size + anom.size
    )
    pooled_std = float(np.sqrt(pooled_var))
    cohens_d = (
        (anom.mean() - normal.mean()) / pooled_std if pooled_std > 0 else float("nan")
    )
    # Overlapping coefficient of the two density histograms on shared bins.
    lo, hi = float(scores.min()), float(scores.max())
    bins = np.linspace(lo, hi, 60)
    hn, _ = np.histogram(normal, bins=bins, density=True)
    ha, _ = np.histogram(anom, bins=bins, density=True)
    widths = np.diff(bins)
    overlap = float(np.sum(np.minimum(hn, ha) * widths))  # 1=identical, 0=disjoint

    higher = "ABOVE" if anom.mean() > normal.mean() else "BELOW"
    print("\ndistribution-overlap diagnostics (gate inputs, not detection metrics):")
    print(f"  direction       : anomalous mean is {higher} normal mean")
    print(
        f"  effect size     : Cohen's d = {cohens_d:+.3f}"
        "   (|d|<0.2 negligible, ~0.5 medium, >0.8 large)"
    )
    print(f"  histogram overlap area = {overlap:.3f}   (1.0 fully overlapping, 0 disjoint)")

    # --- Density-normalized split histograms ---
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(normal, bins=bins, density=True, alpha=0.55, color=NORMAL_COLOR,
            label=f"normal (n={normal.size})")
    ax.hist(anom, bins=bins, density=True, alpha=0.55, color=ANOM_COLOR,
            label=f"anomalous (n={anom.size})")
    ax.axvline(float(normal.mean()), color=NORMAL_COLOR, ls="--", lw=1.2)
    ax.axvline(float(anom.mean()), color=ANOM_COLOR, ls="--", lw=1.2)
    ax.set_xlabel("frame score   (sum |S| over pixels)")
    ax.set_ylabel("density")
    ax.set_title("Ped2 per-frame sparse energy by ground-truth label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"\nsaved density-normalized histogram -> {FIG}")

    # --- Reject gate (preliminary read; confirm from the plot) ---
    inverted = anom.mean() <= normal.mean()
    heavy_overlap = overlap > 0.60 or abs(cohens_d) < 0.30
    print("\n=== REJECT GATE (preliminary read; confirm visually) ===")
    if inverted:
        print("  DIRECTION INVERTED: anomalous frames do NOT carry more sparse energy.")
        print("  -> raw-magnitude reduction REJECTED. Pivot to spatial concentration")
        print("     (nonzero-pixel count / connected-component size) before any metric.")
    elif heavy_overlap:
        print("  HEAVY OVERLAP: distributions are not meaningfully separated.")
        print("  -> raw-magnitude reduction REJECTED. Pivot to spatial concentration")
        print("     (nonzero-pixel count / connected-component size) before any metric.")
    else:
        print("  Measurable separation in the hypothesised direction.")
        print("  -> reduction PROVISIONALLY ACCEPTED; proceed to Step 3 metrics on confirmation.")
    print("  (Detection metrics are intentionally NOT computed yet.)")


if __name__ == "__main__":
    main()
