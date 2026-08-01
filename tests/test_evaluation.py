"""Session 4 logic tests that run WITHOUT the real Ped2 dataset.

The real dataset can't be present in every environment, but the two pieces of
Session-4 logic that must be correct -- (1) label/score alignment across a
multi-clip test set and (2) the Sigma|S| score reduction -- can be proved on a
tiny synthetic Ped2 tree. We fabricate two clips with known ground-truth ranges
and known frame counts, point the config at them, and assert exact behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from rpca_anomaly import config, evaluation
from rpca_anomaly.detection.reduce import frame_sparse_energy


# (clip_name, n_frames, gt_frame_string, expected 1-indexed anomalous frames)
SYNTH_CLIPS = [
    ("Test001", 8, "3:5", {3, 4, 5}),
    ("Test002", 6, "1:2, 6:6", {1, 2, 6}),
]


@pytest.fixture()
def fake_ped2(tmp_path, monkeypatch):
    """Build a minimal UCSDped2/Test tree and repoint config at it."""
    test_root = tmp_path / "UCSDped2" / "Test"
    test_root.mkdir(parents=True)

    gt_lines = []
    for i, (name, n_frames, gt, _anom) in enumerate(SYNTH_CLIPS, start=1):
        clip_dir = test_root / name
        clip_dir.mkdir()
        for f in range(1, n_frames + 1):
            # Distinct 8x8 frames so columns are not degenerate.
            img = np.full((8, 8), f * 3, dtype=np.uint8)
            assert cv2.imwrite(str(clip_dir / f"{f:03d}.tif"), img)
        gt_lines.append(f"TestVideoFile{{{i}}}.gt_frame = [{gt}];")
    (test_root / "UCSDped2.m").write_text("\n".join(gt_lines) + "\n")

    # list_clips("Test") reads config.PED2_TEST; loader resolves UCSDped2.m via
    # the Test dir. Small frame size keeps load_clip / IALM instantaneous.
    monkeypatch.setattr(config, "PED2_TEST", test_root)
    monkeypatch.setattr(config, "TARGET_H", 8)
    monkeypatch.setattr(config, "TARGET_W", 8)
    return test_root


def test_labels_match_ground_truth_ranges(fake_ped2):
    labels, index_map = evaluation.build_labels()

    total = sum(n for _n, n, _g, _a in SYNTH_CLIPS)
    assert labels.shape == (total,)
    assert len(index_map) == total
    assert set(labels.tolist()) <= {0, 1}

    # Walk the index map and confirm every frame's label matches its clip's
    # known 1-indexed anomalous set -- this catches an off-by-one in the
    # 1-indexed-inclusive range expansion.
    expected = {name: anom for name, _n, _g, anom in SYNTH_CLIPS}
    for g, clip_name, local_frame, label in index_map:
        want = 1 if local_frame in expected[clip_name] else 0
        assert label == want, f"{clip_name} frame {local_frame}: got {label}, want {want}"

    # Global indices are contiguous and labels agree with the index map.
    assert [row[0] for row in index_map] == list(range(total))
    assert np.array_equal(labels, np.array([row[3] for row in index_map]))


def test_alignment_holds_and_asserts(fake_ped2):
    labels, index_map = evaluation.build_labels()
    # Alignment against the intended score length (Step 1, before any scores).
    evaluation.assert_alignment(labels, len(labels), index_map)

    # Real scores through the actual pipeline (load -> IALM -> Sigma|S|).
    scores = evaluation.build_scores(cache_dir=fake_ped2.parent / "cache")
    assert scores.shape == labels.shape
    evaluation.assert_alignment(labels, scores, index_map)


def test_score_cache_roundtrip(fake_ped2):
    cache = fake_ped2.parent / "cache"
    first = evaluation.build_scores(cache_dir=cache)
    # Second call must reuse the cache and return identical scores.
    second = evaluation.build_scores(cache_dir=cache)
    np.testing.assert_array_equal(first, second)
    assert (cache / "Test001.npz").exists()


def test_frame_sparse_energy_exact():
    # Column sums of |S|: frame 0 -> 1+2+3=6, frame 1 -> 0, frame 2 -> 4+5=9.
    S = np.array(
        [
            [1.0, 0.0, -4.0],
            [-2.0, 0.0, 5.0],
            [3.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(frame_sparse_energy(S), [6.0, 0.0, 9.0])


def test_frame_sparse_energy_rejects_non_2d():
    with pytest.raises(ValueError):
        frame_sparse_energy(np.zeros(5))


def test_compute_metrics_perfect_separation():
    # Higher score == anomalous, perfectly separable -> ROC-AUC 1.0.
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    m = evaluation.compute_metrics(labels, scores)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc_anom"] == pytest.approx(1.0)
    assert m["prevalence"] == pytest.approx(0.5)
    assert m["f1_best"] == pytest.approx(1.0)


def test_compute_metrics_inverted_scores_below_chance():
    # Higher score tracks NORMAL frames -> ROC-AUC below 0.5, reported straight.
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    m = evaluation.compute_metrics(labels, scores)
    assert m["roc_auc"] < 0.5
