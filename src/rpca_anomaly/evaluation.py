"""Session 4 offline evaluation: pooled frame-level scoring vs. Ped2 GT.

The whole pipeline hangs on one invariant: the per-frame score vector and the
0/1 ground-truth label vector must line up index-for-index across the entire
test set. This module guarantees that by construction -- both vectors are built
by iterating the *same* ordered clip list (`list_clips("Test")`) and, within
each clip, the same frame order (`sorted(glob("*.tif"))`) that the loader and
the ground-truth parser already agree on. `assert_alignment` then proves it
empirically before any metric is computed.

Nothing here is real-time. Clips are decomposed once (cached to `outputs/`) and
scored offline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rpca_anomaly.data.loader import (
    list_clips,
    load_clip,
    load_ground_truth_frames,
)
from rpca_anomaly.detection.reduce import frame_sparse_energy
from rpca_anomaly.solver import rpca_ialm

# index_map rows: (global_frame_index, clip_name, local_frame_1indexed, label)
IndexRow = tuple


def count_frames(clip_dir: Path) -> int:
    """Frame count for a clip without loading pixels.

    Uses the same `sorted(glob("*.tif"))` the loader uses, so this count is
    exactly the `n_frames` `load_clip` would report -- Step 1 can build the
    label vector without paying for image loading or the solver.
    """
    files = sorted(clip_dir.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No .tif frames in {clip_dir}")
    return len(files)


def build_labels(clips: list[Path] | None = None) -> tuple[np.ndarray, list[IndexRow]]:
    """Concatenated 0/1 frame labels over the test set (1 = anomalous).

    Returns the label vector and a parallel index map. Clip order is
    `list_clips("Test")`; the label vector the score vector will be compared
    against is built in exactly that order.
    """
    if clips is None:
        clips = list_clips("Test")

    label_parts: list[np.ndarray] = []
    index_map: list[IndexRow] = []
    g = 0
    for clip_dir in clips:
        n = count_frames(clip_dir)
        clip_labels = load_ground_truth_frames(clip_dir.name, n)
        if len(clip_labels) != n:
            raise ValueError(
                f"{clip_dir.name}: GT length {len(clip_labels)} != frame count {n}"
            )
        for local in range(n):
            index_map.append((g, clip_dir.name, local + 1, int(clip_labels[local])))
            g += 1
        label_parts.append(np.asarray(clip_labels, dtype=int))

    labels = np.concatenate(label_parts).astype(int)
    return labels, index_map


def build_scores(
    clips: list[Path] | None = None,
    cache_dir: str | Path = "outputs",
    verbose: bool = False,
) -> np.ndarray:
    """Pooled per-frame anomaly scores (Sigma|S|) over the test set.

    Decomposes each clip with IALM once, caching X/L/S to `outputs/<clip>.npz`
    (same layout as `scripts/cache_decomposition.py`); reuses the cache on
    later runs. Scores are concatenated in the same `list_clips("Test")` order
    as `build_labels`, so the two vectors are aligned by construction.
    """
    if clips is None:
        clips = list_clips("Test")

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    score_parts: list[np.ndarray] = []
    for clip_dir in clips:
        npz = cache / f"{clip_dir.name}.npz"
        if npz.exists():
            with np.load(npz) as data:
                S = data["S"]
            if verbose:
                print(f"{clip_dir.name}: loaded cached S {S.shape}")
        else:
            clip = load_clip(clip_dir)
            L, S, history = rpca_ialm(clip.X)
            np.savez(npz, X=clip.X, L=L, S=S, shape=np.array(clip.shape))
            if verbose:
                print(
                    f"{clip_dir.name}: decomposed S {S.shape} "
                    f"({len(history)} iters, resid {history[-1]:.2e}), cached"
                )
        score_parts.append(frame_sparse_energy(S))

    return np.concatenate(score_parts)


def assert_alignment(
    labels: np.ndarray,
    scores_or_len,
    index_map: list[IndexRow],
    n_head: int = 5,
    n_tail: int = 5,
) -> None:
    """Prove label/score index-for-index alignment; raise on any mismatch.

    `scores_or_len` may be the score vector or just its length, so Step 1 can
    check alignment against the intended score length before any score exists.
    Prints both lengths, per-clip boundaries, and the first/last
    (frame_index, label) pairs -- the off-by-one evidence.
    """
    n_labels = int(len(labels))
    n_scores = (
        int(scores_or_len)
        if isinstance(scores_or_len, (int, np.integer))
        else int(len(scores_or_len))
    )

    print(f"label vector length : {n_labels}")
    print(f"score vector length : {n_scores}")
    print(f"index_map length    : {len(index_map)}")

    # Per-clip boundaries: where each clip sits in the global index, and how
    # many of its frames are anomalous. This is where an off-by-one in the GT
    # ranges would show up (e.g. an anomaly block starting one frame early).
    print("\nper-clip boundaries (global index is 0-based, inclusive):")
    print(f"  {'clip':<10} {'start':>6} {'end':>6} {'frames':>7} {'anom':>6}")
    clip_first: dict[str, int] = {}
    clip_last: dict[str, int] = {}
    clip_anom: dict[str, int] = {}
    order: list[str] = []
    for g, clip_name, _local, label in index_map:
        if clip_name not in clip_first:
            clip_first[clip_name] = g
            clip_anom[clip_name] = 0
            order.append(clip_name)
        clip_last[clip_name] = g
        clip_anom[clip_name] += int(label)
    total_frames = 0
    total_anom = 0
    for clip_name in order:
        start, end = clip_first[clip_name], clip_last[clip_name]
        nfr = end - start + 1
        nan = clip_anom[clip_name]
        total_frames += nfr
        total_anom += nan
        print(f"  {clip_name:<10} {start:>6} {end:>6} {nfr:>7} {nan:>6}")
    print(f"  {'TOTAL':<10} {'':>6} {'':>6} {total_frames:>7} {total_anom:>6}")

    def pairs(rows: list[IndexRow]) -> str:
        return "  ".join(f"({g}, {label})" for g, _c, _l, label in rows)

    print(f"\nfirst {n_head} (frame_index, label): {pairs(index_map[:n_head])}")
    print(f"last  {n_tail} (frame_index, label): {pairs(index_map[-n_tail:])}")

    # Hard checks -- any failure raises before a metric is ever computed.
    assert n_labels == n_scores, f"LENGTH MISMATCH: labels {n_labels} vs scores {n_scores}"
    assert n_labels == len(index_map), (
        f"LENGTH MISMATCH: labels {n_labels} vs index_map {len(index_map)}"
    )
    assert total_frames == n_labels, (
        f"per-clip frames {total_frames} != label length {n_labels}"
    )
    gidx = np.array([row[0] for row in index_map])
    assert np.array_equal(gidx, np.arange(n_labels)), "index_map global indices not contiguous 0..N-1"
    imap_labels = np.array([row[3] for row in index_map])
    assert np.array_equal(imap_labels, labels), "index_map labels disagree with label vector"

    print("\nALIGNMENT OK")


def split_by_label(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split scores into (normal, anomalous) for the Step-2 histograms."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    return scores[labels == 0], scores[labels == 1]


def compute_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Frame-level detection metrics for a higher-score-is-more-anomalous rule.

    ROC-AUC is the headline. PR-AUC (anomaly class) and best-F1 are reported
    with their honesty baselines: on Ped2 the positive (anomalous) class is the
    *majority*, so a trivial all-positive predictor already scores well --
    `prevalence` and `f1_all_positive` make that explicit, and `pr_auc_normal`
    reports PR-AUC treating the minority normal frames as the positive class.
    No score flipping: if higher energy does not track anomalies, ROC-AUC will
    land below 0.5 and is reported straight.
    """
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
    )

    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = int(labels.size)
    n_anom = int(labels.sum())
    n_normal = n - n_anom
    prevalence = n_anom / n if n else float("nan")

    roc = float(roc_auc_score(labels, scores))
    pr_anom = float(average_precision_score(labels, scores))
    # Minority (normal) class as positive: invert both label and score.
    pr_normal = float(average_precision_score(1 - labels, -scores))

    prec, rec, thr = precision_recall_curve(labels, scores)
    # precision/recall have len(thr)+1 points; the final point has no threshold.
    p, r = prec[:-1], rec[:-1]
    f1s = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    if f1s.size:
        bi = int(np.nanargmax(f1s))
        f1_best = float(f1s[bi])
        threshold = float(thr[bi])
    else:
        f1_best, threshold = float("nan"), float("nan")

    # All-positive baseline: recall 1, precision = prevalence -> F1 = 2p/(p+1).
    f1_all_positive = 2 * prevalence / (prevalence + 1) if n else float("nan")

    return {
        "n_frames": n,
        "n_anom": n_anom,
        "n_normal": n_normal,
        "prevalence": prevalence,
        "roc_auc": roc,
        "pr_auc_anom": pr_anom,
        "pr_auc_normal": pr_normal,
        "f1_best": f1_best,
        "f1_threshold": threshold,
        "f1_all_positive": f1_all_positive,
    }


def _score_bins(scores: np.ndarray, n_bins: int = 60) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    return np.linspace(float(scores.min()), float(scores.max()), n_bins)


def overlap_diagnostics(scores: np.ndarray, labels: np.ndarray, n_bins: int = 60) -> dict:
    """Distribution-overlap diagnostics for the Step-2 reject gate.

    These describe how separated the normal/anomalous score distributions are;
    they are NOT detection metrics. `cohens_d` is the standardized mean gap
    (positive = anomalous higher, the hypothesised direction); `overlap` is the
    overlapping coefficient of the two density histograms (1 = identical,
    0 = disjoint). Shares `_score_bins` with `save_energy_histogram`, so the
    reported overlap matches the drawn plot.
    """
    normal, anom = split_by_label(scores, labels)
    pooled_var = (normal.var() * normal.size + anom.var() * anom.size) / (
        normal.size + anom.size
    )
    pooled_std = float(np.sqrt(pooled_var))
    cohens_d = (
        float((anom.mean() - normal.mean()) / pooled_std)
        if pooled_std > 0
        else float("nan")
    )
    bins = _score_bins(scores, n_bins)
    hn, _ = np.histogram(normal, bins=bins, density=True)
    ha, _ = np.histogram(anom, bins=bins, density=True)
    overlap = float(np.sum(np.minimum(hn, ha) * np.diff(bins)))
    return {
        "normal_mean": float(normal.mean()),
        "anom_mean": float(anom.mean()),
        "direction_ok": bool(anom.mean() > normal.mean()),
        "cohens_d": cohens_d,
        "overlap": overlap,
    }


def save_energy_histogram(
    scores: np.ndarray,
    labels: np.ndarray,
    path: str | Path,
    n_bins: int = 60,
) -> Path:
    """Density-normalized split histogram (normal vs anomalous) -> `path`.

    Density normalization is deliberate: the classes are very imbalanced
    (~1648 anomalous / 362 normal), so raw counts would bury the normal class.
    matplotlib is imported lazily so the pure-logic paths of this module do not
    depend on it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normal, anom = split_by_label(scores, labels)
    bins = _score_bins(scores, n_bins)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(normal, bins=bins, density=True, alpha=0.55, color="#4C78A8",
            label=f"normal (n={normal.size})")
    ax.hist(anom, bins=bins, density=True, alpha=0.55, color="#E45756",
            label=f"anomalous (n={anom.size})")
    ax.axvline(float(normal.mean()), color="#4C78A8", ls="--", lw=1.2)
    ax.axvline(float(anom.mean()), color="#E45756", ls="--", lw=1.2)
    ax.set_xlabel("frame score   (sum |S| over pixels)")
    ax.set_ylabel("density")
    ax.set_title("Ped2 per-frame sparse energy by ground-truth label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
