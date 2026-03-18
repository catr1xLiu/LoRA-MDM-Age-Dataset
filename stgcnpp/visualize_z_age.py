"""
Interactive 3D visualisation of z_age embeddings.

Usage:
  uv run python visualize_z_age.py \\
      --embeddings data/z_age_embeddings_2block_newsplit.npz \\
      --checkpoint checkpoints/vc_age_unfreeze2block_newsplit.pth

Slider picks the starting dimension index d; dims [d, d+1, d+2] map to X/Y/Z.
When a checkpoint is supplied, an arrow is drawn showing the direction of
increasing age — computed as  W[Elderly] − W[Young]  from the final
Linear(32→3) classification layer (head.fc_cls.weight).
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Interactive z_age 3-D viewer")
parser.add_argument(
    "--embeddings", "-e",
    default="data/z_age_embeddings_2block_newsplit.npz",
    help="Path to z_age .npz file (default: 2-block newsplit)",
)
parser.add_argument(
    "--checkpoint", "-c",
    default=None,
    help="Age-classifier .pth checkpoint — used to draw the age-direction arrow",
)
args = parser.parse_args()

# ── Load embeddings ───────────────────────────────────────────────────────────
try:
    d = np.load(args.embeddings)
except FileNotFoundError:
    sys.exit(f"[error] embeddings file not found: {args.embeddings}")

z      = d["z_age"]    # (N, 32)
labels = d["labels"]   # (N,)  0=Young 1=Adult 2=Elderly
splits = d["split"]    # (N,)  "train"/"val"

N, D = z.shape
print(f"Loaded {N} embeddings  ({D}-dim)  from  {args.embeddings}")
print(f"  Young={( labels==0).sum()}  Adult={(labels==1).sum()}  Elderly={(labels==2).sum()}")

# ── Load age-direction vector from checkpoint ─────────────────────────────────
age_vec_32 = None   # 32-dim direction in embedding space
if args.checkpoint:
    try:
        ckpt  = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt))
        W = state["head.fc_cls.weight"].numpy()   # (3, 32)
        # Direction of increasing age:  Elderly row − Young row
        age_vec_32 = W[2] - W[0]                  # (32,)
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"  age-direction norm (32-dim): {np.linalg.norm(age_vec_32):.4f}")
    except Exception as exc:
        print(f"[warning] could not load checkpoint: {exc}")

# ── Constants ─────────────────────────────────────────────────────────────────
COLORS      = ["#1f77b4", "#2ca02c", "#d62728"]
CLASS_NAMES = ["Young", "Adult", "Elderly"]
ARROW_COLOR = "#e377c2"

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(left=0.08, bottom=0.16, right=0.97, top=0.93)

ax3d = fig.add_subplot(111, projection="3d")

ax_slider = fig.add_axes([0.18, 0.06, 0.62, 0.03])
slider = Slider(
    ax_slider,
    "Start dim",
    valmin=0,
    valmax=D - 3,
    valinit=0,
    valstep=1,
    color="#4c78a8",
)

# ── Draw / update ─────────────────────────────────────────────────────────────
def redraw(_=None):
    ax3d.cla()

    d0 = int(slider.val)
    d1, d2 = d0 + 1, d0 + 2

    # Scatter points coloured by age class
    for cls_idx, (name, color) in enumerate(zip(CLASS_NAMES, COLORS)):
        mask = labels == cls_idx
        ax3d.scatter(
            z[mask, d0], z[mask, d1], z[mask, d2],
            c=color, label=f"{name}  (n={mask.sum()})",
            alpha=0.65, s=22, depthshade=True,
        )

    # Age-direction arrow
    if age_vec_32 is not None:
        # Project the 32-dim direction onto the three visible dims
        v = age_vec_32[[d0, d1, d2]]

        # Normalise arrow length to ~25 % of the widest data extent
        chunk  = z[:, [d0, d1, d2]]
        extent = max(chunk.max(axis=0) - chunk.min(axis=0))
        scale  = 0.25 * extent / (np.linalg.norm(v) + 1e-9)
        v_scaled = v * scale

        # Anchor arrow at centroid of all points in these 3 dims
        origin = z[:, [d0, d1, d2]].mean(axis=0)

        start = origin - v_scaled / 2
        end   = origin + v_scaled / 2
        ax3d.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color="#ff0066", linewidth=4.5, solid_capstyle="round",
            label="Age ↑ direction",
        )

    ax3d.set_xlabel(f"Dim {d0}", labelpad=6)
    ax3d.set_ylabel(f"Dim {d1}", labelpad=6)
    ax3d.set_zlabel(f"Dim {d2}", labelpad=6)

    src = args.embeddings.split("/")[-1].replace(".npz", "")
    ax3d.set_title(
        f"{src}  │  dims [{d0}, {d1}, {d2}]",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax3d.legend(loc="upper left", fontsize=9, framealpha=0.85)
    fig.canvas.draw_idle()


slider.on_changed(redraw)
redraw()
plt.show()
