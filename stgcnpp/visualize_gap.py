"""
Visualize the Global Average Pool (GAP) output for the first clip in the dataset.

Produces a 3-panel figure:
  1. 16×16 heatmap of the 256-dim GAP feature vector
  2. Top-10 predicted action class probabilities (softmax bar chart)
  3. Skeleton stick-figure of the first frame of the clip (person 0)
"""

import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Local imports ────────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from stgcnpp import STGCNpp, NTUDataset

# ── NTU-120 action labels (0-indexed) ────────────────────────────────────────
NTU120_LABELS = [
    "drink water","eat meal/snack","brush teeth","brush hair","drop","pickup",
    "throw","sit down","stand up","clapping","reading","writing","tear up paper",
    "wear jacket","take off jacket","wear a shoe","take off a shoe","wear on glasses",
    "take off glasses","put on a hat/cap","take off a hat/cap","cheer up",
    "hand waving","kicking something","reach into pocket","hopping (one foot jumping)",
    "jump up","make a phone call/answer phone","playing with phone/tablet",
    "typing on a keyboard","pointing to something with finger","taking a selfie",
    "check time (from watch)","rub two hands together","nod head/bow","shake head",
    "wipe face","salute","put the palms together","cross hands in front (say stop)",
    "sneeze/cough","staggering","falling","touch head (headache)","touch chest (stomachache/heart pain)",
    "touch back (backache)","touch neck (neckache)","nausea or vomiting condition",
    "use a fan (with hand or paper)/feeling warm","punching/slapping other person",
    "kicking other person","pushing other person","pat on back of other person",
    "point finger at the other person","hugging other person",
    "giving something to other person","touch other person's pocket",
    "handshaking","walking towards each other","walking apart from each other",
    "put on headphone","take off headphone","shoot at the basket","bounce ball",
    "tennis bat swing","juggling table tennis balls","hush (quite)","flick hair",
    "thumb up","thumb down","make ok sign","make victory sign","staple book",
    "counting money","cutting nails","cutting paper (using scissors)",
    "snapping fingers","open bottle","sniff (smell)","squat down","toss a coin",
    "fold paper","clean and wipe glasses","use front fan","apply cream on face",
    "apply cream on hand back","put on bag","take off bag","put something into a bag",
    "take something out of a bag","open a box","move heavy objects","shake fist",
    "throw up cap/hat","hands up (both hands)","cross arms","arm circles",
    "arm swings","running on the spot","butt kicks (kick backward)","cross toe touch",
    "side kick","yawn","stretch oneself","blow nose","hit other person with something",
    "wield knife towards other person","knock over other person (hit with body)",
    "grab other person's stuff","shoot at other person with a gun",
    "step on foot","high-five","cheers and drink","carry something with other person",
    "take a photo of other person","follow other person","whisper in other person's ear",
    "exchange things with other person","support somebody with hand",
    "finger-guessing game (playing rock-paper-scissors)",
]

# ── NTU skeleton edges for stick-figure drawing ──────────────────────────────
NTU_EDGES = [
    (0,1),(1,20),(20,2),(2,3),           # spine
    (20,4),(4,5),(5,6),(6,7),(7,21),(7,22),  # left arm
    (20,8),(8,9),(9,10),(10,11),(11,23),(11,24), # right arm
    (0,16),(16,17),(17,18),(18,19),      # right leg
    (0,12),(12,13),(13,14),(14,15),      # left leg
]

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/ntu120_3danno.pkl"
CKPT_PATH  = "checkpoints/j.pth"
OUT_PATH   = "gap_visualization.png"


def load_checkpoint(model: torch.nn.Module, path: str) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd   = ckpt.get("state_dict", ckpt)
    # remap PYSKL key prefixes
    remapped = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            remapped["backbone." + k[len("backbone."):]] = v
        elif k.startswith("cls_head."):
            remapped["head." + k[len("cls_head."):]] = v
        else:
            remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    # filter out known-skipped keys
    missing = [k for k in missing if not k.startswith("backbone.graph")]
    if missing:
        print(f"  [WARN] Missing keys : {missing[:5]}")
    if unexpected:
        print(f"  [WARN] Unexpected   : {unexpected[:5]}")


def draw_skeleton(ax, joints_3d, title=""):
    """Draw a 2D stick figure using the X/Y plane of 3-D joints."""
    x = joints_3d[:, 0]
    y = joints_3d[:, 1]
    ax.set_aspect("equal")
    # edges
    for (i, j) in NTU_EDGES:
        ax.plot([x[i], x[j]], [y[i], y[j]], color="#4C72B0", lw=1.5, zorder=1)
    # joints
    ax.scatter(x, y, s=30, c="#DD8452", zorder=2, edgecolors="none")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ═════════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load dataset & get the very first clip ─────────────────────────────
    print("Loading dataset…")
    dataset = NTUDataset(
        pkl_path=DATA_PATH,
        split="xsub_val",
        modality="joint",
        clip_len=100,
        num_clips=1,      # just one clip
        test_mode=True,
        num_person=2,
    )
    keypoints_tensor, label = dataset[0]
    # keypoints_tensor: (num_clips=1, M=2, T=100, V=25, C=3)
    # ── also grab raw first-frame for skeleton drawing ──
    # The dataset applies PreNormalize3D; we use the normalised coords for drawing
    first_frame_joints = keypoints_tensor[0, 0, 0].numpy()  # (25, 3) person-0, frame-0

    print(f"  Sample label : {label}  ({NTU120_LABELS[label]})")
    print(f"  Keypoint shape: {tuple(keypoints_tensor.shape)}")

    # ── 2. Build model & load weights ─────────────────────────────────────────
    print("Building model…")
    model = STGCNpp().to(device)
    model.eval()
    load_checkpoint(model, CKPT_PATH)

    # ── 3. Hook the GAP output ────────────────────────────────────────────────
    gap_features = {}

    def hook_fn(module, input, output):
        # output: (N*M, C, 1, 1) with N=1, M=2 → shape (2, C, 1, 1)
        # Average over persons (mirrors what the head does) → (C,)
        gap_features["raw"] = output.detach().cpu().squeeze(-1).squeeze(-1).mean(0)  # (C,)

    hook = model.head.pool.register_forward_hook(hook_fn)

    # ── 4. Forward pass ───────────────────────────────────────────────────────
    # Dataset returns (num_clips, M, T, V, C); model expects (N, M, T, V, C)
    x = keypoints_tensor.to(device)   # (1, 2, 100, 25, 3)

    with torch.no_grad():
        logits = model(x)              # (1, 120)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()  # (120,)

    hook.remove()

    gap_vec = gap_features["raw"].numpy()  # (256,) for person-0 (first in N*M batch)
    print(f"\n  GAP output shape : {gap_vec.shape}")
    print(f"  GAP value range  : [{gap_vec.min():.3f}, {gap_vec.max():.3f}]")
    print(f"  Predicted class  : {probs.argmax()} ({NTU120_LABELS[probs.argmax()]})")
    print(f"  True label       : {label} ({NTU120_LABELS[label]})")

    # ── 5. Visualisation ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5), dpi=150)
    fig.patch.set_facecolor("#F8F8F8")
    gs = GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Panel A: 256-dim GAP heatmap ──────────────────────────────────────────
    ax_gap = fig.add_subplot(gs[0, 0])
    heat   = gap_vec.reshape(16, 16)
    im     = ax_gap.imshow(heat, cmap="RdBu_r", aspect="auto",
                           vmin=-np.abs(heat).max(), vmax=np.abs(heat).max())
    ax_gap.set_title("Global-Average-Pool features\n(256-dim → 16×16 grid)", fontsize=9)
    ax_gap.set_xlabel("Feature index (mod 16)", fontsize=8)
    ax_gap.set_ylabel("Feature index (// 16)",  fontsize=8)
    ax_gap.tick_params(labelsize=7)
    cbar = fig.colorbar(im, ax=ax_gap, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)

    # ── Panel B: Top-10 class probabilities ───────────────────────────────────
    ax_cls = fig.add_subplot(gs[0, 1])
    top10_idx   = np.argsort(probs)[-10:][::-1]
    top10_probs = probs[top10_idx]
    top10_names = [NTU120_LABELS[i] for i in top10_idx]
    colors = ["#2E86AB" if i != label else "#E84855" for i in top10_idx]
    bars = ax_cls.barh(range(10), top10_probs * 100, color=colors, edgecolor="none")
    ax_cls.set_yticks(range(10))
    ax_cls.set_yticklabels(
        [f"{n[:28]}…" if len(n) > 28 else n for n in top10_names],
        fontsize=7
    )
    ax_cls.invert_yaxis()
    ax_cls.set_xlabel("Softmax probability (%)", fontsize=8)
    ax_cls.set_title("Top-10 predicted classes\n(red = ground truth)", fontsize=9)
    ax_cls.tick_params(axis="x", labelsize=7)
    ax_cls.set_xlim(0, max(top10_probs * 100) * 1.15)
    # annotate bars with %
    for bar, p in zip(bars, top10_probs):
        ax_cls.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{p*100:.1f}%", va="center", ha="left", fontsize=7)

    legend_patches = [
        mpatches.Patch(color="#E84855", label="ground truth"),
        mpatches.Patch(color="#2E86AB", label="other classes"),
    ]
    ax_cls.legend(handles=legend_patches, fontsize=7, loc="lower right")

    # ── Panel C: Skeleton stick figure (first frame, person 0) ────────────────
    ax_sk = fig.add_subplot(gs[0, 2])
    draw_skeleton(ax_sk, first_frame_joints,
                  title=f"First frame skeleton\n(person 0, pre-normalised)")

    # ── Super-title ───────────────────────────────────────────────────────────
    pred_label = NTU120_LABELS[probs.argmax()]
    true_label = NTU120_LABELS[label]
    correct    = "✓" if probs.argmax() == label else "✗"
    fig.suptitle(
        f"ST-GCN++  |  xsub_val[0]  |  "
        f"GT: «{true_label}»  →  Pred: «{pred_label}»  {correct}",
        fontsize=10, fontweight="bold", y=1.01
    )

    plt.savefig(OUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
