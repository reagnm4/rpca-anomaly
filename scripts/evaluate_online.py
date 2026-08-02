"""Checkpoint C (Session 5): causal online-RPCA evaluation on Ped2 vs. batch.

Learns the normal subspace ONCE from the training clips (all normal), then
scores every test frame causally with the online detector -- V1 (frozen U) and
V2 (GROUSE-adapted U) -- and prints an ablation table against the Session-4
batch baseline. Same labels, same Sigma|S| reduction, same alignment proof as
the batch pipeline (build_labels / assert_alignment / compute_metrics /
frame_sparse_energy are reused verbatim).

Honest expectation: online should land at or BELOW batch (ROC-AUC 0.628); it
forfeits the non-causal advantage of seeing every frame. If it exceeds batch,
suspect a leak and re-check tests/test_online.py first.

    python scripts/evaluate_online.py            # full run (few min on a laptop)
    python scripts/evaluate_online.py --no-check-clean   # skip the RPCA check

Reported r is chosen by 95%-energy AND the run is repeated at r=20; the
pedestrians-still-visible check compares plain vs. RPCA-cleaned subspaces so we
can see whether cleaning strips normal pedestrians into the sparse part.
"""
from __future__ import annotations

import argparse

import numpy as np

from rpca_anomaly.data.loader import list_clips, load_clip
from rpca_anomaly.detection.reduce import frame_sparse_energy
from rpca_anomaly.evaluation import assert_alignment, build_labels, compute_metrics
from rpca_anomaly.online import (
    learn_normal_model,
    per_frame_timing,
    run_online,
    select_rank,
)
from rpca_anomaly.solver import rpca_ialm

# Session-4 batch reference (README "Results (Session 4)").
BATCH_ROC = 0.628
BATCH_PRN = 0.417
PRN_BASELINE = 0.180  # 1 - prevalence; PR-AUC-normal chance level


def stack_split(split: str):
    clips = list_clips(split)
    mats = [load_clip(c) for c in clips]
    frames = np.concatenate([m.X for m in mats], axis=1)
    return clips, mats, frames


def pedestrians_visible_check(train_frames, energy, subsample, seed=0):
    """Do normal pedestrians survive RPCA cleaning, or go into the sparse part?

    Runs batch RPCA on a training subsample and compares the 95%-energy rank of
    the raw vs. cleaned (low-rank) matrix. If cleaning collapses the rank and
    moves most mass to S, the cleaned subspace is background-only -> plain U is
    the right source. Cheap: uses a subsample, not all training frames.
    """
    rng = np.random.default_rng(seed)
    N = train_frames.shape[1]
    idx = np.sort(rng.choice(N, min(subsample, N), replace=False))
    sub = train_frames[:, idx]

    mu = sub.mean(axis=1, keepdims=True)
    _, sv_plain, _ = np.linalg.svd(sub - mu, full_matrices=False)
    r95_plain, _ = select_rank(sv_plain, energy)

    print(f"  running batch RPCA on {sub.shape[1]} training frames (one-time)...")
    L, S, _ = rpca_ialm(sub)
    _, sv_clean, _ = np.linalg.svd(L - L.mean(axis=1, keepdims=True), full_matrices=False)
    r95_clean, _ = select_rank(sv_clean, energy)
    sparse_mass = float(np.abs(S).sum() / (np.abs(sub - mu).sum() + 1e-12))

    print(f"  r@{energy:.0%} energy  plain={r95_plain}   cleaned-L={r95_clean}")
    print(f"  sparse-mass fraction (moved to S by RPCA) = {sparse_mass:.3f}")
    stripped = (r95_clean < 0.5 * r95_plain) or (sparse_mass > 0.30)
    if stripped:
        print("  -> cleaning pushes pedestrians into S; cleaned L is ~background-only.")
        print("     RECOMMENDATION: use plain U (normal pedestrian variation retained).")
    else:
        print("  -> pedestrians largely survive in L; cleaned U is a viable alternative.")
    return "plain" if stripped else "clean"


def score_test_causal(mats, model, adapt, verify_reduction=False):
    """Concatenated causal scores over test clips, in list_clips('Test') order."""
    parts = []
    for i, m in enumerate(mats):
        if verify_reduction and i == 0:
            sc, _bg, sp = run_online(m.X, model, adapt=adapt, return_components=True)
            # Prove the online reduction is byte-identical to the batch one.
            assert np.allclose(sc, frame_sparse_energy(sp)), "reduction mismatch"
        else:
            sc = run_online(m.X, model, adapt=adapt)
        parts.append(sc)
    return np.concatenate(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--energy", type=float, default=0.95)
    ap.add_argument("--target-sparsity", type=float, default=0.10)
    ap.add_argument("--k", type=float, default=3.0)
    ap.add_argument("--inner-iters", type=int, default=3)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--u-source", choices=["plain", "clean"], default="plain")
    ap.add_argument("--no-check-clean", dest="check_clean", action="store_false")
    ap.add_argument("--clean-subsample", type=int, default=300)
    args = ap.parse_args()

    labels, index_map = build_labels()
    train_clips, _train_mats, train_frames = stack_split("Train")
    print(f"train: {len(train_clips)} clips, {train_frames.shape[1]} frames, m={train_frames.shape[0]}")
    test_clips, test_mats, _ = stack_split("Test")
    print(f"test : {len(test_clips)} clips, {sum(m.n_frames for m in test_mats)} frames\n")

    if args.check_clean:
        print("pedestrians-still-visible check:")
        rec = pedestrians_visible_check(train_frames, args.energy, args.clean_subsample)
        print(f"  (using --u-source {args.u_source}; check recommends: {rec})\n")

    clean = args.u_source == "clean"

    # Model at r@energy, then repeat at r=20.
    model95 = learn_normal_model(
        train_frames, energy=args.energy, clean=clean,
        target_sparsity=args.target_sparsity, k=args.k, inner_iters=args.inner_iters,
    )
    r95 = model95.r
    configs = [(f"r@{args.energy:.0%}={r95}", model95)]
    if r95 != 20:
        model20 = learn_normal_model(
            train_frames, fixed_r=20, clean=clean,
            target_sparsity=args.target_sparsity, k=args.k, inner_iters=args.inner_iters,
        )
        configs.append(("r=20", model20))

    d = model95.diagnostics
    print(f"model: r@{args.energy:.0%}={r95}  lambda={model95.lam:.4g}  tau={model95.tau:.4g}  "
          f"train-sparsity={d['achieved_sparsity']:.3f} (target {args.target_sparsity})\n")

    timing_clip = test_mats[0].X  # representative clip for latency
    rows = []
    aligned_once = False
    for rlabel, model in configs:
        for adapt, vname in [(False, "V1 (frozen U)"), (True, "V2 (GROUSE)")]:
            scores = score_test_causal(test_mats, model, adapt, verify_reduction=not aligned_once)
            if not aligned_once:
                assert_alignment(labels, scores, index_map)  # prove online scores align
                print()
                aligned_once = True
            else:
                assert len(scores) == len(labels)
            met = compute_metrics(labels, scores)
            tim = per_frame_timing(timing_clip, model, adapt=adapt, inner_iters=args.inner_iters,
                                   eta=args.eta, warmup=5)
            rows.append((vname, rlabel, met["roc_auc"], met["pr_auc_normal"],
                         tim["mean_ms"], tim["p95_ms"], tim["fps"]))

    # ---- ablation table ----
    print("=" * 84)
    print("  Ped2 frame-level: online (causal) vs batch (non-causal)   reduction: sum|S|/frame")
    print("=" * 84)
    print(f"  {'config':<18} {'r':<10} {'ROC-AUC':>8} {'PR-AUC(norm)':>13} "
          f"{'mean ms':>8} {'95p ms':>7} {'FPS':>6}")
    print("-" * 84)
    print(f"  {'batch (S4)':<18} {'-':<10} {BATCH_ROC:>8.3f} {BATCH_PRN:>13.3f} "
          f"{'-':>8} {'-':>7} {'-':>6}   non-causal")
    for vname, rlabel, roc, prn, mms, p95, fps in rows:
        print(f"  {vname:<18} {rlabel:<10} {roc:>8.3f} {prn:>13.3f} "
              f"{mms:>8.3f} {p95:>7.3f} {fps:>6.0f}")
    print("-" * 84)
    print(f"  PR-AUC(normal) chance baseline = {PRN_BASELINE:.3f}   |   Ped2 10 FPS, webcam ~30 FPS")
    print("=" * 84)


if __name__ == "__main__":
    main()
