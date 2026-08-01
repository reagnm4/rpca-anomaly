"""Session 5 online-RPCA logic tests -- run WITHOUT the real Ped2 dataset.

The two claims of the session live or die here: CAUSALITY (a score never depends
on a future frame) and the correctness of the GROUSE subspace update. Both are
proved on small synthetic low-rank + sparse sequences.
"""
from __future__ import annotations

import numpy as np
import pytest

from rpca_anomaly.online import (
    NormalModel,
    calibrate_lambda,
    grouse_update,
    infer_frame,
    learn_normal_model,
    run_online,
    select_rank,
)


def _orthonormal(m: int, r: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.linalg.qr(rng.standard_normal((m, r)))[0][:, :r]


def _synthetic_sequence(m=120, r=6, T=40, seed=1):
    """Frames from a fixed subspace + noise + occasional sparse spikes.

    Returns (frames (m,T), model). The last frame carries a large spike so a
    (leaky) global normalization would depend on the whole sequence.
    """
    rng = np.random.default_rng(seed)
    U = _orthonormal(m, r, seed)
    mu = rng.standard_normal(m)
    W = rng.standard_normal((r, T))
    frames = mu[:, None] + U @ W + 0.05 * rng.standard_normal((m, T))
    # sparse spikes on a few frames; a big one on the last frame
    for t in (7, 19, T - 1):
        idx = rng.choice(m, size=m // 8, replace=False)
        frames[idx, t] += (5.0 if t == T - 1 else 1.5)
    lam = 0.1
    model_scores = run_online(frames, NormalModel(mu, U, lam, np.inf, r))
    tau = float(np.median(model_scores))  # gate ~half the frames for V2 exercise
    return frames, NormalModel(mu, U, lam, tau, r)


# --------------------------------------------------------------------------- #
# CAUSALITY -- the gate for everything downstream
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("adapt", [False, True], ids=["V1", "V2"])
def test_causality_prefix_equals_full(adapt):
    """score(t) from running on [0..t] must EQUAL score from running on [0..T]."""
    frames, model = _synthetic_sequence()
    full = run_online(frames, model, adapt=adapt)
    for t in range(frames.shape[1]):
        pref = run_online(frames[:, : t + 1], model, adapt=adapt)
        assert pref[t] == full[t], f"future leaked into score({t}) [adapt={adapt}]"


@pytest.mark.parametrize("adapt", [False, True], ids=["V1", "V2"])
def test_leak_test_has_teeth(adapt):
    """A deliberately non-causal score (global min-max normalization) MUST be
    caught by the same prefix-vs-full comparison -- otherwise the test is
    vacuous."""
    frames, model = _synthetic_sequence()

    def leaky(seq):
        s = run_online(seq, model, adapt=adapt)
        return (s - s.min()) / (s.max() - s.min() + 1e-12)  # peeks at global extremes

    full = leaky(frames)
    caught = False
    for t in range(frames.shape[1] - 1):  # exclude last (holds the global max)
        pref = leaky(frames[:, : t + 1])
        if pref[t] != full[t]:
            caught = True
            break
    assert caught, "leak test failed to catch a known future-dependent score"


# --------------------------------------------------------------------------- #
# GROUSE update
# --------------------------------------------------------------------------- #
def test_grouse_preserves_orthonormality():
    m, r = 120, 6
    U = _orthonormal(m, r, seed=3)
    rng = np.random.default_rng(4)
    w = rng.standard_normal(r)
    v = rng.standard_normal(m)
    res = v - U @ (U.T @ v)  # orthogonal to range(U), as in run_online
    U2 = grouse_update(U, w, res, eta=0.1)
    np.testing.assert_allclose(U2.T @ U2, np.eye(r), atol=1e-9)


def test_grouse_moves_subspace_toward_residual():
    m, r = 60, 3
    U = _orthonormal(m, r, seed=5)
    rng = np.random.default_rng(6)
    w = rng.standard_normal(r)
    v = rng.standard_normal(m)
    res = v - U @ (U.T @ v)
    before = np.linalg.norm(res - U @ (U.T @ res))       # res is fully outside U
    U2 = grouse_update(U, w, res, eta=0.3)
    after = np.linalg.norm(res - U2 @ (U2.T @ res))       # should be captured more
    assert after < before


# --------------------------------------------------------------------------- #
# Offline training model
# --------------------------------------------------------------------------- #
def test_select_rank_energy():
    sv = np.array([10.0, 1.0, 0.1, 0.01])  # variance ~ sv**2, first dominates
    r, cum = select_rank(sv, energy=0.95)
    assert r == 1
    assert cum[-1] == pytest.approx(1.0)


def test_learn_model_recovers_subspace():
    # Low noise -> a clean 5-D signal, so r@95%-energy finds the planted rank.
    m, r_true, N = 150, 5, 300
    rng = np.random.default_rng(7)
    U_true = _orthonormal(m, r_true, seed=8)
    mu = rng.standard_normal(m)
    W = rng.standard_normal((r_true, N))
    train = mu[:, None] + U_true @ W + 0.02 * rng.standard_normal((m, N))

    model = learn_normal_model(train, energy=0.95, target_sparsity=0.10, calib_max_frames=N)

    assert model.diagnostics["r95_plain"] <= r_true + 2
    captured = np.linalg.norm(model.U @ (model.U.T @ U_true)) / np.linalg.norm(U_true)
    assert captured > 0.95
    assert model.tau > model.diagnostics["train_score_mean"]


def test_calibrate_lambda_hits_target_sparsity():
    # Substantial residual (noise orthogonal to the r=5 subspace) so a target
    # active-pixel fraction is actually reachable by thresholding.
    m, r_true, N = 150, 5, 300
    rng = np.random.default_rng(12)
    U_true = _orthonormal(m, r_true, seed=13)
    mu = rng.standard_normal(m)
    W = rng.standard_normal((r_true, N))
    train = mu[:, None] + U_true @ W + 0.15 * rng.standard_normal((m, N))

    model = learn_normal_model(
        train, fixed_r=r_true, target_sparsity=0.10, k=3.0, calib_max_frames=N
    )
    assert model.lam > 0
    assert abs(model.diagnostics["achieved_sparsity"] - 0.10) < 0.05

    # calibrate_lambda is monotone: a larger lambda yields sparser residuals.
    sp_hi = model.diagnostics["achieved_sparsity"]
    from rpca_anomaly.online import _mean_sparsity

    sp_higher_lam = _mean_sparsity(train, model.mu, model.U, model.lam * 2, inner_iters=3)
    assert sp_higher_lam < sp_hi


def test_fixed_r_overrides_energy():
    m, N = 100, 120
    rng = np.random.default_rng(9)
    train = rng.standard_normal((m, N))
    model = learn_normal_model(train, fixed_r=20, calib_max_frames=N)
    assert model.r == 20
    assert model.U.shape == (m, 20)


def test_infer_frame_reconstructs_normal_frame():
    # A frame that lies exactly in the subspace should leave ~no sparse residual.
    m, r = 80, 4
    U = _orthonormal(m, r, seed=10)
    rng = np.random.default_rng(11)
    mu = rng.standard_normal(m)
    w_true = rng.standard_normal(r)
    x = mu + U @ w_true
    bg, s, w = infer_frame(x, mu, U, lam=0.05, inner_iters=3)
    assert np.abs(s).sum() < 1e-6
    np.testing.assert_allclose(bg, x, atol=1e-6)
