"""Run with: pytest tests/ -v  (after implementing loader.py TODOs).

These pin the Session-1 contract so Sessions 2-3 can trust X blindly.
"""
import numpy as np
import pytest

from rpca_anomaly import config
from rpca_anomaly.data import loader


@pytest.fixture(scope="module")
def clip():
    clips = loader.list_clips("Test")
    assert len(clips) == 12, "Ped2 has exactly 12 test clips"
    assert not any(c.name.endswith("_gt") for c in clips), "_gt dirs leaked in"
    return loader.load_clip(clips[0])


def test_matrix_shape(clip):
    m = config.TARGET_H * config.TARGET_W
    assert clip.X.shape == (m, clip.n_frames)
    assert clip.X.dtype == np.dtype(config.X_DTYPE)


def test_value_range(clip):
    assert clip.X.min() >= 0.0 and clip.X.max() <= 1.0
    assert clip.X.max() > 0.1, "all-near-zero suggests a scaling bug"


def test_column_frame_roundtrip(clip):
    """Column j reshaped to (h, w) must equal preprocessed frame j."""
    h, w = clip.shape
    col0 = clip.X[:, 0].reshape(h, w)
    # Re-load frame 0 independently through the same pipeline:
    frame_files = sorted(loader.list_clips("Test")[0].glob("*.tif"))
    f0 = loader.preprocess_frame(loader.load_frame(frame_files[0]))
    np.testing.assert_allclose(col0, f0)


def test_columns_are_distinct(clip):
    """Frames differ over time — identical columns means an indexing bug."""
    assert not np.allclose(clip.X[:, 0], clip.X[:, -1])
