"""Session 4 score reduction: sparse matrix S -> one scalar per frame.

Reduction rule (fixed for this session): the anomaly score of frame t is the
total sparse energy in that frame,

    score(t) = sum over pixels of |S[:, t]|   (L1 norm of column t of S)

Rationale: an anomalous frame is hypothesised to push more mass into the sparse
component than ordinary pedestrian motion does. This is *raw* energy with no
normalization: every Ped2 frame has the same pixel count (TARGET_H * TARGET_W),
so the raw sum is directly comparable across frames and across clips, which is
exactly what pooling the 12 test clips into one score vector requires. Per-clip
min-max normalization (as in the Session-3 `scoring.py`) would force every
clip onto [0, 1] independently and destroy that comparability, so it is
deliberately not used here.
"""
from __future__ import annotations

import numpy as np


def frame_sparse_energy(S: np.ndarray) -> np.ndarray:
    """Total sparse energy per frame.

    Parameters
    ----------
    S : np.ndarray, shape (m_pixels, n_frames)
        Sparse component of one clip's decomposition X = L + S.

    Returns
    -------
    np.ndarray, shape (n_frames,)
        score(t) = sum_i |S[i, t]| for each frame t. Raw, un-normalized.
    """
    S = np.asarray(S)
    if S.ndim != 2:
        raise ValueError(f"S must be 2-D (pixels, frames); got shape {S.shape}")
    return np.abs(S).sum(axis=0)
