"""Central config — paths and preprocessing constants.

Keep every magic number here so Sessions 2-4 never hardcode anything.
"""
from pathlib import Path

# Repo root = two levels up from this file's package dir
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PED2_ROOT = DATA_DIR / "UCSD_Anomaly_Dataset.v1p2" / "UCSDped2"
PED2_TRAIN = PED2_ROOT / "Train"
PED2_TEST = PED2_ROOT / "Test"
OUTPUT_DIR = REPO_ROOT / "outputs"

# Preprocessing (see project doc §5)
# Native Ped2 frames are 240x360 (h, w). Downsample so the SVD inside
# IALM stays cheap: m = TARGET_H * TARGET_W is the tall dimension of X.
TARGET_H = 120
TARGET_W = 160

# dtype for X. TODO(Reagan): decide float32 vs float64 and note why —
# consider what the IALM SVD costs at m ≈ 19,200 and what precision the
# nuclear-norm / soft-thresholding steps actually need.
X_DTYPE = "float64"
