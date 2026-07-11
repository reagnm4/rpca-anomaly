# Robust PCA Video Anomaly Detection

Unsupervised anomaly detection in fixed-camera surveillance video via
low-rank + sparse decomposition (Robust PCA / Principal Component Pursuit),
solved with Inexact Augmented Lagrange Multipliers (IALM).

Each video is stacked as a matrix **X ∈ R^(m×n)** (m = pixels/frame,
n = frames) and decomposed as **X = L + S**, where L is the low-rank
background and S is the sparse foreground. Anomalies are scored per-frame
from the sparse component.

**Dataset:** UCSD Pedestrian (Ped2) — fixed camera, grayscale, ships with
frame-level and pixel-level ground-truth anomaly labels.

## Status

- [X] Session 1 — Repo, data loading, preprocessing, X-matrix construction
- [ ] Session 2 — IALM / PCP core (SVT, low-rank + sparse decomposition)
- [ ] Session 3 — Detection pipeline (heatmaps, temporal anomaly scores, viz)
- [ ] Session 4 — Evaluation vs PCA baseline, ROC-AUC/F1 on Ped2, demo

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_ucsd.sh   # ~700MB, extracts to data/
```

## Layout

```
src/rpca_anomaly/
  data/       # Ped2 loading, preprocessing, frame->matrix construction
  rpca/       # IALM solver, singular value thresholding   (Session 2)
  detection/  # anomaly scoring, thresholding               (Session 3)
  viz/        # 4-panel demo, heatmaps, score plots         (Session 3)
scripts/      # dataset download, run entrypoints
notebooks/    # exploration only — logic lives in src/
data/         # gitignored; see data/README.md
```
