import os
import numpy as np

from rpca_anomaly.data.loader import list_clips, load_clip
from rpca_anomaly.solver import rpca_ialm

OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

clip = load_clip(list_clips("Test")[0])          # Test001
print(f"{clip.clip_name}: X shape {clip.X.shape}, {clip.n_frames} frames")

L, S, history = rpca_ialm(clip.X, verbose=True)
print(f"converged in {len(history)} iterations, final residual {history[-1]:.2e}")

assert S.shape == clip.X.shape
assert S.shape[1] == clip.n_frames

path = f"{OUT}/{clip.clip_name}.npz"
np.savez(path, X=clip.X, L=L, S=S, shape=np.array(clip.shape))
print(f"cached to {path} ({os.path.getsize(path) / 1e6:.0f} MB)")