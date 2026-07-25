import numpy as np


def score_frames(S, normalize=True):
    scores = np.linalg.norm(S, axis=0) 

    if normalize:
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-12:
            return np.zeros_like(scores)
        scores = (scores - lo) / (hi - lo)

    return scores