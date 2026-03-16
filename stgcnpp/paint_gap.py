"""
Paint the Global Average Pool for the first clip — two views:
  Left  : 2-D painted heatmap (16×16 grid, colour = activation)
  Right : 3-D bar forest      (height + colour both = activation)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (side-effect import)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from stgcnpp import STGCNpp, NTUDataset

DATA_PATH = "data/ntu120_3danno.pkl"
CKPT_PATH = "checkpoints/j.pth"
OUT_PATH  = "gap_painted.png"


# ── checkpoint loader ─────────────────────────────────────────────────────────
def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd   = ckpt.get("state_dict", ckpt)
    remapped = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            remapped[k] = v
        elif k.startswith("cls_head."):
            remapped["head." + k[len("cls_head."):]] = v
        else:
            remapped[k] = v
    model.load_state_dict(remapped, strict=False)


# ── run inference and capture GAP ─────────────────────────────────────────────
def get_gap_vector():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NTUDataset(
        pkl_path=DATA_PATH, split="xsub_val", modality="joint",
        clip_len=100, num_clips=1, test_mode=True, num_person=2,
    )
    keypoints, label = dataset[0]          # (1, 2, 100, 25, 3)

    model = STGCNpp().to(device).eval()
    load_checkpoint(model, CKPT_PATH)

    captured = {}
    def hook(module, inp, out):
        # out: (N*M, C, 1, 1)  →  mean over persons → (C,)
        captured["gap"] = out.detach().cpu().squeeze(-1).squeeze(-1).mean(0).numpy()

    h = model.head.pool.register_forward_hook(hook)
    with torch.no_grad():
        logits = model(keypoints.to(device))
    h.remove()

    probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    return captured["gap"], label, probs.argmax()


# ── helpers ───────────────────────────────────────────────────────────────────
NTU120_LABELS = [
    "drink water","eat meal/snack","brush teeth","brush hair","drop","pickup",
    "throw","sit down","stand up","clapping","reading","writing","tear up paper",
    "wear jacket","take off jacket","wear a shoe","take off a shoe","wear on glasses",
    "take off glasses","put on a hat/cap","take off a hat/cap","cheer up",
    "hand waving","kicking something","reach into pocket","hopping","jump up",
    "make a phone call","playing with phone/tablet","typing on a keyboard",
    "pointing to something","taking a selfie","check time (from watch)",
    "rub two hands together","nod head/bow","shake head","wipe face","salute",
    "put the palms together","cross hands in front","sneeze/cough","staggering",
    "falling","touch head","touch chest","touch back","touch neck","nausea/vomiting",
    "use a fan","punching other person","kicking other person","pushing other person",
    "pat on back","point finger at other","hugging other person",
    "giving something to other","touch other's pocket","handshaking",
    "walking towards each other","walking apart","put on headphone",
    "take off headphone","shoot at basket","bounce ball","tennis bat swing",
    "juggling table tennis balls","hush","flick hair","thumb up","thumb down",
    "make ok sign","make victory sign","staple book","counting money",
    "cutting nails","cutting paper","snapping fingers","open bottle","sniff",
    "squat down","toss a coin","fold paper","clean glasses","use front fan",
    "apply cream on face","apply cream on hand","put on bag","take off bag",
    "put something into a bag","take something out of bag","open a box",
    "move heavy objects","shake fist","throw up cap","hands up","cross arms",
    "arm circles","arm swings","running on the spot","butt kicks","cross toe touch",
    "side kick","yawn","stretch oneself","blow nose","hit other person",
    "wield knife","knock over other person","grab other's stuff",
    "shoot other with gun","step on foot","high-five","cheers and drink",
    "carry something with other","take a photo of other","follow other person",
    "whisper in other's ear","exchange things","support somebody",
    "rock-paper-scissors",
]

CMAP = plt.cm.YlOrRd           # light theme: white → yellow → orange → red

# text colours for light background
TC  = "#1a1a1a"   # titles / labels
GC  = "#cccccc"   # grid lines
PC  = "#555555"   # pane edges


def bar3d_colored(ax, grid, cmap):
    """
    Draw a 3-D bar for every cell in `grid` (16×16).
    Both height and face colour are proportional to the normalised activation.
    """
    nrows, ncols = grid.shape
    norm_grid    = grid / grid.max()           # [0, 1]

    dx = dy = 0.75
    for row in range(nrows):
        for col in range(ncols):
            z    = norm_grid[row, col]
            rgba = cmap(z)
            ax.bar3d(
                col - dx / 2, row - dy / 2, 0,
                dx, dy, z,
                color=rgba, shade=True, zsort="average",
                edgecolor="none",
            )

    ax.set_xlim(-1, ncols)
    ax.set_ylim(-1, nrows)
    ax.set_zlim(0, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([0, 0.5, 1.0])
    ax.tick_params(axis="z", labelsize=7, pad=2, colors=TC)
    ax.set_zlabel("norm. activation", fontsize=8, labelpad=6, color=TC)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(PC)
    ax.yaxis.pane.set_edgecolor(PC)
    ax.zaxis.pane.set_edgecolor(PC)
    ax.grid(False)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("Running inference…")
    gap, label, pred = get_gap_vector()

    grid      = gap.reshape(16, 16)
    norm_grid = grid / grid.max()
    gt_name   = NTU120_LABELS[label]
    pred_name = NTU120_LABELS[pred]
    correct   = "✓" if pred == label else "✗"

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6), dpi=160)
    fig.patch.set_alpha(0)                     # transparent background

    ax2d = fig.add_axes([0.03, 0.08, 0.40, 0.82])
    ax3d = fig.add_axes([0.48, 0.04, 0.50, 0.92], projection="3d")

    # ── 2-D painted heatmap ───────────────────────────────────────────────────
    im = ax2d.imshow(norm_grid, cmap=CMAP, vmin=0, vmax=1,
                     interpolation="nearest", aspect="equal")

    # annotate top-20 % neurons
    thresh = np.percentile(norm_grid, 80)
    for r in range(16):
        for c in range(16):
            v = norm_grid[r, c]
            if v >= thresh:
                # dark text on bright cells, light text on pale cells
                txt_color = "white" if v > 0.65 else "#333333"
                ax2d.text(c, r, f"{v:.2f}", ha="center", va="center",
                          fontsize=4.5, color=txt_color, fontweight="bold")

    ax2d.set_xticks(np.arange(-0.5, 16, 1), minor=True)
    ax2d.set_yticks(np.arange(-0.5, 16, 1), minor=True)
    ax2d.grid(which="minor", color=GC, linewidth=0.4)
    ax2d.tick_params(which="both", bottom=False, left=False,
                     labelbottom=False, labelleft=False)
    ax2d.set_title("2-D activation map\n(256 neurons → 16 × 16 grid)",
                   color=TC, fontsize=10, pad=10)
    ax2d.set_facecolor("none")                 # transparent axes background

    cbar = fig.colorbar(im, ax=ax2d, orientation="vertical",
                        fraction=0.046, pad=0.02)
    cbar.set_label("normalised activation", color=TC, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color=TC, labelsize=7)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TC)
    cbar.outline.set_edgecolor(GC)

    # ── 3-D bar forest ────────────────────────────────────────────────────────
    ax3d.set_facecolor("none")                 # transparent pane background
    bar3d_colored(ax3d, norm_grid, CMAP)
    ax3d.set_title("3-D activation forest\n(height + colour = firing strength)",
                   color=TC, fontsize=10, pad=8)
    ax3d.tick_params(colors=TC)
    ax3d.zaxis.label.set_color(TC)
    ax3d.view_init(elev=28, azim=-50)

    # ── super-title ───────────────────────────────────────────────────────────
    fig.suptitle(
        f"Global Average Pool  ·  xsub_val[0]  ·  "
        f"GT: «{gt_name}»  →  Pred: «{pred_name}» {correct}",
        color=TC, fontsize=11, fontweight="bold", y=0.99,
    )

    plt.savefig(OUT_PATH, bbox_inches="tight",
                transparent=True, dpi=160)
    print(f"Saved → {OUT_PATH}")
    print(f"  GAP shape    : {gap.shape}")
    print(f"  Value range  : [{gap.min():.3f}, {gap.max():.3f}]")
    print(f"  Active (>0.5 norm): {(norm_grid > 0.5).sum()} / 256 neurons")
    print(f"  Label: {gt_name}  |  Pred: {pred_name}  {correct}")


if __name__ == "__main__":
    main()
