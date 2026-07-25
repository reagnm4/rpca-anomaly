from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rpca_anomaly import config
import cv2
import re


@dataclass
class ClipMatrix:
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


def load_ground_truth_frames(clip_name: str, n_frames: int) -> np.ndarray:
    m_path = list_clips("Test")[0].parent / "UCSDped2.m"
    blocks = re.findall(r"gt_frame\s*=\s*\[([^\]]+)\]", m_path.read_text())

    idx = int(clip_name.removeprefix("Test")) - 1
    if not 0 <= idx < len(blocks):
        raise ValueError(f"{clip_name}: no gt entry (found {len(blocks)} clips)")

    labels = np.zeros(n_frames, dtype=int)
    for part in blocks[idx].split(","):
        start, end = (int(v) for v in part.split(":"))
        labels[start - 1:end] = 1
    return labels
