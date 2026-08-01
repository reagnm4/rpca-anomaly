"""Step 3 (Session 4): the one-command reproducible frame-level evaluation.

Regenerates every Session-4 number from a single command:

    python scripts/evaluate_ped2.py

It builds the label vector and the Sigma|S| score vector (reusing the outputs/
decomposition cache), proves alignment, saves the density-normalized histogram,
and prints the metrics table. ROC-AUC is the headline (comparability with
published Ped2 results). PR-AUC and F1 are printed next to their majority-class
baselines: Ped2's frame-level split is ~82% anomalous, so a trivial all-positive
predictor already scores PR-AUC ~= prevalence and F1 ~= 0.90 -- the minority
(normal-class) PR-AUC is the honest imbalance metric. Nothing is flipped: if the
raw-energy reduction does not track anomalies, ROC-AUC lands below 0.5 and is
reported as-is.
"""
from __future__ import annotations

from rpca_anomaly.evaluation import (
    assert_alignment,
    build_labels,
    build_scores,
    compute_metrics,
    overlap_diagnostics,
    save_energy_histogram,
)

FIG = "figures/ped2_energy_hist.png"


def _row(metric: str, value: float, baseline: float, note: str) -> str:
    return f"  {metric:<26} {value:>7.3f}   {baseline:>8.3f}   {note}"


def main() -> None:
    labels, index_map = build_labels()
    scores = build_scores(verbose=True)
    print()
    assert_alignment(labels, scores, index_map)
    save_energy_histogram(scores, labels, FIG)

    m = compute_metrics(labels, scores)
    diag = overlap_diagnostics(scores, labels)

    print("\n" + "=" * 72)
    print("  Ped2 frame-level evaluation  (Session 4)")
    print("=" * 72)
    print("  reduction : score(t) = sum|S| over pixels  (raw sparse energy / frame)")
    print(
        f"  frames    : {m['n_frames']}   "
        f"(anomalous {m['n_anom']} / normal {m['n_normal']}, "
        f"prevalence {m['prevalence']:.3f})"
    )
    print("-" * 72)
    print(f"  {'metric':<26} {'value':>7}   {'baseline':>8}   note")
    print("-" * 72)
    print(_row("ROC-AUC", m["roc_auc"], 0.5, "headline; chance = 0.5"))
    print(_row("PR-AUC (anomalous=pos)", m["pr_auc_anom"], m["prevalence"],
               "baseline = prevalence"))
    print(_row("PR-AUC (normal=pos)", m["pr_auc_normal"], 1 - m["prevalence"],
               "minority class; honest metric"))
    print(_row("F1 (best threshold)", m["f1_best"], m["f1_all_positive"],
               f"@ thr={m['f1_threshold']:.4g}; baseline = all-positive"))
    print("-" * 72)
    print(
        f"  gate diagnostics: Cohen's d = {diag['cohens_d']:+.3f}, "
        f"histogram overlap = {diag['overlap']:.3f}  (not detection metrics)"
    )
    print(f"  histogram -> {FIG}")
    print("=" * 72)

    verdict = "ABOVE" if m["roc_auc"] > 0.5 else "AT/BELOW"
    print(
        f"\nROC-AUC {m['roc_auc']:.3f} is {verdict} chance. "
        "Reported straight; see README for interpretation."
    )


if __name__ == "__main__":
    main()
