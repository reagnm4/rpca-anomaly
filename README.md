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
- [X] Session 2 — IALM / PCP core (SVT, low-rank + sparse decomposition)
- [X] Session 3 — Detection pipeline (heatmaps, temporal anomaly scores, viz)
- [ ] Session 4 — Evaluation vs PCA baseline, ROC-AUC/F1 on Ped2, demo

## Results (Session 3)

Frame-level scoring on Test001: L2 norm of each column of S.

| metric | value | baseline |
|---|---|---|
| ROC-AUC | 0.122 | 0.500 |
| PR-AUC | 0.484 | 0.667 |

Below chance in both. The score ranks anomalous frames below normal ones, and raw norms span only 7.19 to 8.64 against a mean of 7.62, so the detector has almost no dynamic range. Min-max normalization stretches that thin band across the full axis, which makes the plot look more separated than the underlying scores are.

Foreground pixel count correlates with score at only 0.318, so this is not
simply crowd density. Measured on one clip; whether the inversion is a
property of the method or specific to Test001 is open until all twelve
test clips are scored.

![scores vs ground truth](figures/test001_scores_gt.png)
![sparse component frames](figures/test001_S_frames.png)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_ucsd.sh   # ~700MB, extracts to data/
pip install -e .
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
