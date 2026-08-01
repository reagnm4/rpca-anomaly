"""Online, causal RPCA anomaly detection (GRASTA family).

This is NOT batch RPCA on a sliding window. A normal low-rank subspace U is
learned ONCE offline from the Ped2 *training* clips (all normal). At test time
each frame is processed causally in constant time by alternating a projection
onto U with a soft-threshold that absorbs the anomalous, sparse residual:

    x_t = mu + U w_t + s_t ,   score(t) = sum |s_t|     (same reduction as batch)

Two variants:
  * V1 -- frozen subspace: U is fixed after training.
  * V2 -- adaptive subspace: after each frame that scores as normal, U is nudged
    by one GROUSE (Grassmannian) rank-one step so it tracks slow normal drift.
    The update is GATED on normalcy (score <= tau) so anomalies are never
    absorbed into the "normal" model. This pair (fixed vs. GROUSE-tracked
    subspace with an L1 robust cost) is the GRASTA algorithm.

Causality is the whole point and is enforced by construction: nothing here ever
looks past frame t (no future frame, no global normalization). `run_online`
over a prefix [0..t] yields byte-identical score(t) to running over [0..T] --
proved by the leak test in tests/test_online.py.

Per-frame cost is O(m*r): two mat-vecs with U plus a shrink, independent of t.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from rpca_anomaly.solver import rpca_ialm, shrink


@dataclass
class NormalModel:
    """The normal-appearance model learned offline from training frames."""

    mu: np.ndarray            # (m,) mean training frame
    U: np.ndarray             # (m, r) orthonormal normal subspace
    lam: float                # soft-threshold on the sparse residual
    tau: float                # normalcy threshold (train score mean + k*std)
    r: int
    diagnostics: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Per-frame causal inference
# --------------------------------------------------------------------------- #
def infer_frame(
    x: np.ndarray,
    mu: np.ndarray,
    U: np.ndarray,
    lam: float,
    inner_iters: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Robustly explain one frame with the fixed subspace U.

    Returns (background, s, w): background = mu + U w is the recovered normal
    frame, s is the sparse anomalous residual, w are the subspace coordinates.
    Works in centered coordinates (x - mu); mu is added back only for the
    returned background image.
    """
    xc = x - mu
    s = np.zeros_like(xc)
    lc = np.zeros_like(xc)
    w = np.zeros(U.shape[1])
    for _ in range(inner_iters):
        w = U.T @ (xc - s)          # O(m*r) least-squares coordinates
        lc = U @ w                  # O(m*r) low-rank (normal) part, centered
        s = shrink(xc - lc, lam)    # O(m)   sparse anomalous residual
    return mu + lc, s, w


def frame_score(s: np.ndarray) -> float:
    """score = sum|s| over pixels -- identical reduction to the batch pipeline
    (`detection.reduce.frame_sparse_energy` applied to one column)."""
    return float(np.abs(s).sum())


# --------------------------------------------------------------------------- #
# GROUSE subspace update (V2)
# --------------------------------------------------------------------------- #
def grouse_update(U: np.ndarray, w: np.ndarray, res: np.ndarray, eta: float) -> np.ndarray:
    """One GROUSE rank-one geodesic step on the Grassmannian.

    `res` is the fit residual on the sparse-corrected target and is orthogonal
    to range(U) by construction (res = (I - U U^T)(x - mu - s)); `p = U w` lies
    in range(U). The update rotates the column space in the plane they span, so
    orthonormality is preserved to machine precision -- no re-orthonormalization
    needed. O(m*r).
    """
    nw = float(np.linalg.norm(w))
    nres = float(np.linalg.norm(res))
    p = U @ w
    npn = float(np.linalg.norm(p))
    if nw < 1e-12 or nres < 1e-12 or npn < 1e-12:
        return U
    sigma = nres * nw
    step = ((np.cos(eta * sigma) - 1.0) * (p / npn) + np.sin(eta * sigma) * (res / nres))
    return U + np.outer(step, w / nw)


# --------------------------------------------------------------------------- #
# Causal run over a sequence (one clip)
# --------------------------------------------------------------------------- #
def run_online(
    frames: np.ndarray,
    model: NormalModel,
    adapt: bool = False,
    eta: float = 0.1,
    inner_iters: int = 3,
    reorth_every: int = 0,
    return_components: bool = False,
):
    """Score a clip causally, frame by frame, left to right.

    frames : (m, T) -- columns are frames, matching the X = L + S layout.
    adapt  : False -> V1 (frozen U); True -> V2 (GROUSE update, gated on score<=tau).
    U is reset to model.U at entry, so each clip is an independent causal stream.
    Returns scores (T,), or (scores, backgrounds, sparses) if return_components.
    """
    m, T = frames.shape
    U = model.U.copy()
    mu, lam, tau = model.mu, model.lam, model.tau

    scores = np.empty(T)
    bgs = np.empty((m, T)) if return_components else None
    sps = np.empty((m, T)) if return_components else None

    for t in range(T):
        x = frames[:, t]
        bg, s, w = infer_frame(x, mu, U, lam, inner_iters)
        scores[t] = frame_score(s)
        if return_components:
            bgs[:, t] = bg
            sps[:, t] = s
        if adapt and scores[t] <= tau:
            # Gated GROUSE step. res = (x - mu - s) - U w is orthogonal to U.
            target = (x - mu) - s
            res = target - (U @ w)
            U = grouse_update(U, w, res, eta)
            if reorth_every and (t + 1) % reorth_every == 0:
                U, _ = np.linalg.qr(U)

    if return_components:
        return scores, bgs, sps
    return scores


# --------------------------------------------------------------------------- #
# Offline training: learn the normal model
# --------------------------------------------------------------------------- #
def select_rank(singular_values: np.ndarray, energy: float = 0.95) -> tuple[int, np.ndarray]:
    """Smallest r whose singular values capture >= `energy` of total variance."""
    ev = np.asarray(singular_values, dtype=float) ** 2
    cum = np.cumsum(ev) / ev.sum()
    r = int(np.searchsorted(cum, energy) + 1)
    return max(r, 1), cum


def _mean_sparsity(
    frames: np.ndarray, mu: np.ndarray, U: np.ndarray, lam: float, inner_iters: int
) -> float:
    """Mean fraction of nonzero pixels in the sparse residual over frames."""
    m, N = frames.shape
    total = 0.0
    for t in range(N):
        _, s, _ = infer_frame(frames[:, t], mu, U, lam, inner_iters)
        total += np.count_nonzero(s) / m
    return total / N


def calibrate_lambda(
    frames: np.ndarray,
    mu: np.ndarray,
    U: np.ndarray,
    target_sparsity: float = 0.10,
    inner_iters: int = 3,
    lo: float = 1e-4,
    hi: float = 1.0,
    steps: int = 18,
) -> float:
    """Pick lambda so normal training frames hit `target_sparsity` active pixels.

    Sparsity decreases monotonically in lambda, so geometric bisection converges.
    Calibrated on TRAINING frames only -- never touches test data.
    """
    for _ in range(steps):
        mid = float(np.sqrt(lo * hi))
        sp = _mean_sparsity(frames, mu, U, mid, inner_iters)
        if sp > target_sparsity:   # too many nonzeros -> raise lambda
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def learn_normal_model(
    train_frames: np.ndarray,
    energy: float = 0.95,
    fixed_r: int | None = None,
    clean: bool = False,
    target_sparsity: float = 0.10,
    k: float = 3.0,
    inner_iters: int = 3,
    calib_max_frames: int = 400,
    rng: np.random.Generator | None = None,
) -> NormalModel:
    """Learn (mu, U, lambda, tau) from normal training frames.

    train_frames : (m, N) -- all-normal training columns.
    energy       : cumulative-variance target for r@energy (reported always).
    fixed_r      : if given, overrides the energy-selected rank (e.g. r=20).
    clean        : if True, U comes from an RPCA-cleaned low-rank part L of the
                   training matrix instead of the raw centered matrix. The
                   diagnostics record r@energy for BOTH so the caller can run
                   the "are pedestrians still visible in L?" check.
    k            : tau = train_score.mean() + k * train_score.std().
    """
    mu = train_frames.mean(axis=1)
    Xc = train_frames - mu[:, None]

    # r@energy on the raw (plain) centered training matrix, always reported.
    _, sv_plain, _ = np.linalg.svd(Xc, full_matrices=False)
    r95_plain, cum_plain = select_rank(sv_plain, energy)

    diagnostics: dict = {
        "energy": energy,
        "r95_plain": r95_plain,
        "cum_energy_plain": cum_plain,
        "clean": clean,
    }

    if clean:
        # Offline, one-time. RPCA typically pushes moving pedestrians into S, so
        # L may collapse to background -> tiny r95. That is exactly the check.
        L, S, _ = rpca_ialm(train_frames)
        Lc = L - L.mean(axis=1)[:, None]
        U_all, sv_clean, _ = np.linalg.svd(Lc, full_matrices=False)
        r95_clean, cum_clean = select_rank(sv_clean, energy)
        diagnostics["r95_clean"] = r95_clean
        diagnostics["cum_energy_clean"] = cum_clean
        # How much mass RPCA moved to the sparse part (pedestrians -> S).
        diagnostics["sparse_mass_fraction"] = float(
            np.abs(S).sum() / (np.abs(train_frames - mu[:, None]).sum() + 1e-12)
        )
        singular = sv_clean
        r_source = r95_clean
    else:
        U_all = np.linalg.svd(Xc, full_matrices=False)[0]
        singular = sv_plain
        r_source = r95_plain

    r = int(fixed_r) if fixed_r is not None else r_source
    r = max(1, min(r, U_all.shape[1]))
    U = np.ascontiguousarray(U_all[:, :r])
    diagnostics["r_used"] = r

    # Calibrate lambda on a subsample of training frames (offline).
    if train_frames.shape[1] > calib_max_frames:
        rng = rng or np.random.default_rng(0)
        idx = np.sort(rng.choice(train_frames.shape[1], calib_max_frames, replace=False))
        calib_frames = train_frames[:, idx]
    else:
        calib_frames = train_frames
    lam = calibrate_lambda(calib_frames, mu, U, target_sparsity, inner_iters)
    diagnostics["target_sparsity"] = target_sparsity
    diagnostics["achieved_sparsity"] = _mean_sparsity(calib_frames, mu, U, lam, inner_iters)

    # tau from the training-frame score distribution (V1 inference, frozen U).
    tmp = NormalModel(mu=mu, U=U, lam=lam, tau=np.inf, r=r)
    train_scores = run_online(train_frames, tmp, adapt=False, inner_iters=inner_iters)
    tau = float(train_scores.mean() + k * train_scores.std())
    diagnostics["train_score_mean"] = float(train_scores.mean())
    diagnostics["train_score_std"] = float(train_scores.std())
    diagnostics["k"] = k

    return NormalModel(mu=mu, U=U, lam=lam, tau=tau, r=r, diagnostics=diagnostics)


# --------------------------------------------------------------------------- #
# Real-time timing harness
# --------------------------------------------------------------------------- #
def per_frame_timing(
    frames: np.ndarray,
    model: NormalModel,
    adapt: bool = False,
    eta: float = 0.1,
    inner_iters: int = 3,
    warmup: int = 5,
) -> dict:
    """Measure per-frame inference latency (compute only, no I/O).

    Times each frame's inference (and, for V2, the gated update) with
    perf_counter, discards `warmup` frames, and reports mean/95p ms and the
    effective FPS = 1000 / mean_ms.
    """
    m, T = frames.shape
    U = model.U.copy()
    mu, lam, tau = model.mu, model.lam, model.tau
    times_ms: list[float] = []

    for t in range(T):
        x = frames[:, t]
        t0 = perf_counter()
        bg, s, w = infer_frame(x, mu, U, lam, inner_iters)
        score = frame_score(s)
        if adapt and score <= tau:
            target = (x - mu) - s
            res = target - (U @ w)
            U = grouse_update(U, w, res, eta)
        dt = (perf_counter() - t0) * 1e3
        if t >= warmup:
            times_ms.append(dt)

    arr = np.asarray(times_ms) if times_ms else np.asarray([float("nan")])
    mean_ms = float(arr.mean())
    return {
        "n_timed": int(arr.size),
        "mean_ms": mean_ms,
        "p95_ms": float(np.percentile(arr, 95)),
        "fps": float(1000.0 / mean_ms) if mean_ms > 0 else float("inf"),
    }
