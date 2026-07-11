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
import cv2


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
    root = config.PED2_TEST if split == "Test" else config.PED2_TRAIN
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and not d.name.endswith("_gt")
    )


def load_frame(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cv2 could not read frame: {path}")
    return img.astype(np.float64) / 255.0


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(
        frame,
        (config.TARGET_W, config.TARGET_H),
        interpolation=cv2.INTER_AREA,
    )


def frames_to_matrix(frames: np.ndarray) -> np.ndarray:
    n = frames.shape[0]
    return frames.reshape(n, -1).T



def load_clip(clip_dir: Path) -> ClipMatrix:
    files = sorted(clip_dir.glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No .tif frames in {clip_dir}")
    frames = np.stack([preprocess_frame(load_frame(f)) for f in files])
    X = frames_to_matrix(frames).astype(config.X_DTYPE)
    return ClipMatrix(
        X=X,
        shape=(config.TARGET_H, config.TARGET_W),
        clip_name=clip_dir.name,
        n_frames=len(files),
    )


def load_ground_truth_frames(clip_name: str) -> np.ndarray:
    """Frame-level anomaly labels (0/1 per frame) parsed from UCSDped2.m.

    Needed in Session 4 for ROC-AUC / F1 — stub is fine today.
    """
    # TODO(Session 4)
    raise NotImplementedError
