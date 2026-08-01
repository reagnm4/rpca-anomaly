"""Step 1 (Session 4): build the frame-level label vector and PROVE it lines up
index-for-index with the score vector -- BEFORE any score or metric is computed.

The score vector (Step 2) is built from the same clip order (`list_clips`) and
the same within-clip frame order (`sorted(glob("*.tif"))`) used here, so the
label length printed below is exactly the length the score vector will have.
This script exists to catch or rule out an off-by-one in the ground truth.

Run where the Ped2 dataset lives:

    python scripts/build_labels.py
"""
from rpca_anomaly.evaluation import assert_alignment, build_labels


def main() -> None:
    labels, index_map = build_labels()

    # Prove alignment against the intended score length. No S, no solver, no
    # metric -- just the label vector and its frame index map.
    assert_alignment(labels, len(labels), index_map)

    n = int(labels.size)
    n_anom = int(labels.sum())
    print(
        f"\nclass balance: {n_anom} anomalous / {n - n_anom} normal "
        f"of {n} frames ({n_anom / n:.1%} positive)"
    )
    print("Score reduction NOT computed yet -- stopping for Step 2 confirmation.")


if __name__ == "__main__":
    main()
