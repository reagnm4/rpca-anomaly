import matplotlib.pyplot as plt
import numpy as np
from rpca_anomaly.data.loader import list_clips, load_clip
from rpca_anomaly.solver import rpca_ialm

clip = load_clip(list_clips("Test")[0])   # Test001
print(f"{clip.clip_name}: X shape {clip.X.shape}, {clip.n_frames} frames")

L, S, history = rpca_ialm(clip.X, verbose=True)
print(f"converged in {len(history)} iterations, final residual {history[-1]:.2e}")

frame = 100
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, mat, title in zip(axes, [clip.X, L, S], ["Original", "L (Background)", "S (Sparse Foreground)"]):
    ax.imshow(mat[:, frame].reshape(clip.shape), cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.savefig("demo_frame.png", dpi=150)
plt.show()