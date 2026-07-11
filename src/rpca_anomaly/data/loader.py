"""Ped2 loading, preprocessing, and X-matrix construction.

Session 1 core — implement the TODOs below. Everything downstream
(IALM in Session 2, heatmaps in Session 3) consumes what this module
produces, so lock the contract now:

    X ∈ R^(m x n),  m = TARGET_H * TARGET_W pixels,  n = frames,
    column j = frame j flattened; values in [0, 1].

Ped2 ships as ordered grayscale TIFFs per clip (001.tif, 002.tif, ...),
not video files — "frame extraction" = sorted load, not decoding.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rpca_anomaly import config


@dataclass
class ClipMatrix:
    """One clip, ready for RPCA.

    Attributes:
        X: (m, n) matrix, one flattened frame per column.
        shape: (h, w) of the preprocessed frames — REQUIRED to reshape
            columns of S back into 2D anomaly heatmaps in Session 3.
        clip_name: e.g. "Test001".
        n_frames: number of frames (== X.shape[1]).
    """
    X: np.ndarray
    shape: tuple[int, int]
    clip_name: str
    n_frames: int


def list_clips(split: str = "Test") -> list[Path]:
    """Return sorted clip directories for a split ("Train" or "Test").

    Must EXCLUDE the *_gt ground-truth mask folders that sit alongside
    the Test clips.
    """
    # TODO(Reagan): implement. Hint: the _gt dirs and real clip dirs
    # differ only by suffix — decide how you'll filter robustly.
    raise NotImplementedError


def load_frame(path: Path) -> np.ndarray:
    """Load one TIFF as grayscale float in [0, 1] at native resolution."""
    # TODO(Reagan): cv2.imread with the right flag, then dtype/scale.
    # Watch out: cv2.imread returns None (not an exception) on failure.
    raise NotImplementedError


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Downsample to (TARGET_H, TARGET_W).

    Choice to justify: which cv2 interpolation flag for *shrinking*
    an image, and why the default is the wrong one here.
    """
    # TODO(Reagan)
    raise NotImplementedError


def frames_to_matrix(frames: np.ndarray) -> np.ndarray:
    """Stack (n, h, w) preprocessed frames into X ∈ R^(m, n).

    Contract: column j of X is frame j flattened. Be deliberate about
    reshape order — the SAME order must invert cleanly when Session 3
    reshapes columns of S back to (h, w). Add a round-trip test.
    """
    # TODO(Reagan)
    raise NotImplementedError


def load_clip(clip_dir: Path) -> ClipMatrix:
    """Full pipeline for one clip: list TIFFs -> load -> preprocess -> X."""
    # TODO(Reagan): compose the functions above. Sort frame files
    # numerically (lexicographic sort breaks if names lack zero-padding —
    # Ped2 is zero-padded, but don't rely on it silently).
    raise NotImplementedError


def load_ground_truth_frames(clip_name: str) -> np.ndarray:
    """Frame-level anomaly labels (0/1 per frame) parsed from UCSDped2.m.

    Needed in Session 4 for ROC-AUC / F1 — stub is fine today.
    """
    # TODO(Session 4)
    raise NotImplementedError
