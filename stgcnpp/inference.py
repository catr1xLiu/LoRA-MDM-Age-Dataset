"""
ST-GCN++ single clip inference script with visualization.

Loads a checkpoint and runs inference on a single clip from either the NTU
RGB+D action-recognition dataset or the Van Criekinge age-recognition dataset,
then displays an interactive visualization.

Usage
-----
    # Joint modality
    uv run python inference.py \\
        --checkpoint checkpoints/j.pth \\
        --split xsub_val \\
        --clip 1

    # Bone modality (visualization shows joints, not bone vectors)
    uv run python inference.py \\
        --checkpoint checkpoints/b.pth \\
        --modality bone \\
        --split xsub_val \\
        --clip 1

Output
------
    Displays an interactive visualization with:
    - 3D skeleton animation (100 frames, looping) -- always shows joints
    - Draggable progress bar
    - Top-10 predictions as horizontal bar chart
    - True label marked

Note
----
    For bone modality, inference uses bone features (edge vectors between joints),
    but the visualization always displays the skeleton using joint positions
    for intuitive interpretation.
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.widgets import Button, Slider

from stgcnpp import AgeClassifierHead, STGCNpp, build_dataloader
from stgcnpp.dataset import pre_normalize_3d, uniform_sample_frames

# fmt: off
NTU_ACTION_NAMES = {
    # --- Daily Actions (A1–A40, A61–A102) ---
    0: "drink water",           1: "eat meal",              2: "brush teeth",
    3: "brush hair",            4: "drop",                  5: "pick up",
    6: "throw",                 7: "sit down",              8: "stand up",
    9: "clapping",              10: "reading",              11: "writing",
    12: "tear up paper",        13: "put on jacket",        14: "take off jacket",
    15: "put on a shoe",        16: "take off a shoe",      17: "put on glasses",
    18: "take off glasses",     19: "put on a hat/cap",     20: "take off a hat/cap",
    21: "cheer up",             22: "hand waving",          23: "kicking something",
    24: "reach into pocket",    25: "hopping",              26: "jump up",
    27: "phone call",           28: "play with phone",      29: "type on keyboard",
    30: "point to something",   31: "taking a selfie",      32: "check time (watch)",
    33: "rub two hands",        34: "nod head/bow",         35: "shake head",
    36: "wipe face",            37: "salute",               38: "put palms together",
    39: "cross hands in front",
    # --- Medical Conditions (A41–A49) ---
    40: "sneeze/cough",         41: "staggering",           42: "falling down",
    43: "headache",             44: "chest pain",           45: "back pain",
    46: "neck pain",            47: "nausea/vomiting",      48: "fan self",
    # --- Mutual Actions (A50–A60) ---
    49: "punch/slap",           50: "kicking",              51: "pushing",
    52: "pat on back",          53: "point finger",         54: "hugging",
    55: "giving object",        56: "touch pocket",         57: "shaking hands",
    58: "walking towards",      59: "walking apart",
    # --- Daily Actions cont. (A61–A102) ---
    60: "put on headphone",     61: "take off headphone",   62: "shoot at basket",
    63: "bounce ball",          64: "tennis bat swing",     65: "juggle table tennis ball",
    66: "hush",                 67: "flick hair",           68: "thumb up",
    69: "thumb down",           70: "make OK sign",         71: "make victory sign",
    72: "staple book",          73: "counting money",       74: "cutting nails",
    75: "cutting paper",        76: "snap fingers",         77: "open bottle",
    78: "sniff/smell",          79: "squat down",           80: "toss a coin",
    81: "fold paper",           82: "ball up paper",        83: "play magic cube",
    84: "apply cream on face",  85: "apply cream on hand",  86: "put on bag",
    87: "take off bag",         88: "put object into bag",  89: "take object out of bag",
    90: "open a box",           91: "move heavy objects",   92: "shake fist",
    93: "throw up cap/hat",     94: "capitulate",           95: "cross arms",
    96: "arm circles",          97: "arm swings",           98: "run on the spot",
    99: "butt kicks",           100: "cross toe touch",    101: "side kick",
    # --- Medical Conditions cont. (A103–A105) ---
    102: "yawn",                103: "stretch oneself",     104: "blow nose",
    # --- Mutual Actions cont. (A106–A120) ---
    105: "hit with object",     106: "wield knife",         107: "knock over",
    108: "grab stuff",          109: "shoot with gun",      110: "step on foot",
    111: "high-five",           112: "cheers and drink",    113: "carry object",
    114: "take a photo",        115: "follow",              116: "whisper",
    117: "exchange things",     118: "support somebody",    119: "rock-paper-scissors",
}
# fmt: on

AGE_GROUP_NAMES = {
    0: "Young (<40)",
    1: "Adult (40-64)",
    2: "Elderly (>=65)",
}


_NTU_BONE_PAIRS = (
    (0, 1),
    (1, 20),
    (2, 20),
    (3, 2),
    (4, 20),
    (5, 4),
    (6, 5),
    (7, 6),
    (8, 20),
    (9, 8),
    (10, 9),
    (11, 10),
    (12, 0),
    (13, 12),
    (14, 13),
    (15, 14),
    (16, 0),
    (17, 16),
    (18, 17),
    (19, 18),
    (21, 22),
    (20, 20),
    (22, 7),
    (23, 24),
    (24, 11),
)

_NTU_JOINT_COLORS = [
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "magenta",
    "pink",
    "brown",
    "gray",
    "olive",
    "lime",
    "teal",
    "navy",
    "maroon",
    "coral",
    "gold",
    "indigo",
    "violet",
    "black",
    "salmon",
    "khaki",
    "plum",
    "tan",
]


def _remap_key(key: str) -> str | None:
    """Map a PYSKL checkpoint key to our model's key."""
    if key.startswith("backbone."):
        return key
    if key.startswith("head."):
        return key
    if key.startswith("cls_head."):
        return "head." + key[len("cls_head.") :]
    return None


def load_checkpoint(
    model: STGCNpp, state_dict: dict[str, torch.Tensor]
) -> tuple[list[str], list[str], list[str]]:
    """Load a checkpoint state dict into our model."""
    remapped: dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for k, v in state_dict.items():
        new_k = _remap_key(k)
        if new_k is None:
            skipped.append(k)
        else:
            remapped[new_k] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    return missing, unexpected, skipped


def load_checkpoint_raw(ckpt_path: str, device: torch.device) -> tuple[dict, dict[str, torch.Tensor]]:
    """Load raw checkpoint object and normalized state dict."""
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    return raw, raw.get("state_dict", raw)


def detect_task_from_state_dict(state_dict: dict[str, torch.Tensor]) -> str:
    """Detect whether the checkpoint is for action or age recognition."""
    head_keys = {
        key
        for key in state_dict
        if key.startswith("head.") or key.startswith("cls_head.")
    }

    has_bottleneck = any(key.endswith(".fc_z.weight") for key in head_keys)
    has_classifier = any(key.endswith(".fc_cls.weight") for key in head_keys)

    if has_bottleneck and has_classifier:
        return "age"
    if has_classifier:
        return "action"
    raise RuntimeError("Could not detect model type from classifier head layers")


def build_model_for_task(task: str) -> STGCNpp:
    """Instantiate a model matching the checkpoint task."""
    if task == "age":
        model = STGCNpp(in_channels=3, num_classes=120)
        model.head = AgeClassifierHead(
            in_channels=256, z_dim=32, num_classes=3, dropout=0.3
        )
        return model
    if task == "action":
        return STGCNpp(in_channels=3, num_classes=120)
    raise ValueError(f"Unsupported task: {task}")


def resolve_age_years(frame_dir: str, ages_root: Path) -> float | None:
    """Resolve per-clip age in years from HumanML3D age text files."""
    try:
        subject_dir, trial_id = frame_dir.split("_", 1)
    except ValueError:
        return None

    age_file = ages_root / subject_dir / f"{trial_id}_humanml3d_22joints.txt"
    if not age_file.exists():
        return None

    text = age_file.read_text(encoding="utf-8").strip()
    if not text:
        return None

    try:
        return float(text)
    except ValueError:
        return None


def label_name(task: str, label: int) -> str:
    """Get a human-readable class name for the current task."""
    if task == "age":
        return AGE_GROUP_NAMES.get(label, f"Unknown ({label})")
    return NTU_ACTION_NAMES.get(label, f"Unknown ({label})")


def load_annotation(pkl_path: str) -> dict:
    """Load the NTU annotation pickle file."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def get_sample_info(annotation: dict, split: str, index: int) -> dict:
    """Get sample information by index from a split."""
    split_data = annotation["split"].get(split, [])
    if index >= len(split_data):
        raise IndexError(
            f"Index {index} out of range for split '{split}' (length {len(split_data)})"
        )

    frame_dir = split_data[index]
    ann_by_id = {ann["frame_dir"]: ann for ann in annotation["annotations"]}
    sample = ann_by_id[frame_dir]

    return {
        "index": index,
        "frame_dir": frame_dir,
        "label": sample["label"],
        "total_frames": sample["total_frames"],
        "keypoint": sample["keypoint"],
    }


def prepare_visualization_joints(
    raw_keypoint: np.ndarray,
    clip_len: int = 100,
    num_clips: int = 10,
) -> np.ndarray:
    """Process raw keypoint for visualization (always uses joint modality).

    Args:
        raw_keypoint: Raw keypoint array (M, T, 25, 3) from annotation.
        clip_len:     Number of frames per clip.
        num_clips:    Number of clips (uses first clip for visualization).

    Returns:
        Processed keypoint array (num_clips, M, clip_len, V, C) for visualization.
    """
    keypoint = raw_keypoint.astype(np.float32)

    keypoint = pre_normalize_3d(keypoint)
    total_frames = keypoint.shape[1]

    frame_inds = uniform_sample_frames(
        total_frames, clip_len, num_clips, test_mode=True
    )
    keypoint = keypoint[:, frame_inds, :, :]

    M, T_total, V, C = keypoint.shape
    keypoint = keypoint.reshape(M, num_clips, clip_len, V, C)
    keypoint = keypoint.transpose(1, 0, 2, 3, 4)

    return np.ascontiguousarray(keypoint)


@torch.no_grad()
def inference_single(
    model: STGCNpp,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    sample_index: int,
) -> tuple[int, torch.Tensor, np.ndarray, np.ndarray]:
    """Run inference on a single sample.

    Args:
        model:        The STGCNpp model in eval mode.
        dataloader:   Test DataLoader yielding (keypoint, label) batches.
        device:       Inference device.
        sample_index: Index of the sample to inference on.

    Returns:
        Tuple of (true_label, probs_tensor, keypoint_numpy, gap_vector)
    """
    model.eval()

    dataset = dataloader.dataset

    if sample_index >= len(dataset):
        raise IndexError(
            f"Sample index {sample_index} out of range (dataset has {len(dataset)} samples)"
        )

    keypoint, label = dataset[sample_index]
    keypoint_np = keypoint.numpy()

    keypoint = keypoint.unsqueeze(0).to(device, non_blocking=True)

    B, NC, M, T, V, C = keypoint.shape

    keypoint = keypoint.view(B * NC, M, T, V, C)

    captured = {}

    def hook(module, inp, out):
        captured["gap"] = out.detach().cpu().squeeze(-1).squeeze(-1).mean(0).numpy()

    h = model.head.pool.register_forward_hook(hook)

    logits = model(keypoint)

    h.remove()

    probs = F.softmax(logits, dim=1)

    gap = captured.get("gap", np.zeros(256))

    return label, probs, keypoint_np, gap


_GAP_CMAP = plt.cm.YlOrRd


def bar3d_colored(ax, grid, cmap):
    """Draw a 3-D bar for every cell in `grid` (16×16).

    Both height and face colour are proportional to the normalised activation.
    """
    nrows, ncols = grid.shape
    norm_grid = grid / grid.max() if grid.max() > 0 else grid

    dx = dy = 0.75
    for row in range(nrows):
        for col in range(ncols):
            z = norm_grid[row, col]
            rgba = cmap(z)
            ax.bar3d(
                col - dx / 2,
                row - dy / 2,
                0,
                dx,
                dy,
                z,
                color=rgba,
                shade=True,
                zsort="average",
                edgecolor="none",
            )

    ax.set_xlim(-1, ncols)
    ax.set_ylim(-1, nrows)
    ax.set_zlim(0, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([0, 0.5, 1.0])
    ax.tick_params(axis="z", labelsize=7, pad=2)
    ax.set_zlabel("norm. activation", fontsize=8, labelpad=6)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)


def create_visualization(
    keypoint: np.ndarray,
    task: str,
    true_label: int,
    probs: np.ndarray,
    sample_info: dict,
    gap: np.ndarray,
    fps: int = 30,
):
    """Create interactive visualization with 3D skeleton and predictions.

    Args:
        keypoint:     Skeleton data shape (num_clips, M, clip_len, V, C)
        true_label:   Ground truth label
        probs:        Predicted probabilities shape (num_clips, num_classes)
        sample_info:  Sample information dict
        gap:          Global average pool vector (256,)
        fps:          Animation frames per second

    Returns:
        tuple: (figure, animation)
    """
    NC, M, T, V, C = keypoint.shape
    gap_elev = 45
    gap_spin_deg_per_frame = 2.0

    keypoint_clip = keypoint[0, 0]
    num_frames = T

    probs_avg = probs.mean(axis=0)
    num_classes = probs_avg.shape[0]
    topk = min(10, num_classes)
    topk_probs, topk_indices = torch.from_numpy(probs_avg).topk(topk)
    topk_probs = topk_probs.numpy()
    topk_indices = topk_indices.numpy()

    fig = plt.figure(figsize=(14, 8))
    title_suffix = ""
    if sample_info.get("age_years") is not None:
        title_suffix = f" | age {sample_info['age_years']:.1f} years"
    fig.suptitle(
        f"ST-GCN++ {task.title()} Inference: {sample_info['frame_dir']}{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[2, 1],
        height_ratios=[1, 1],
        left=0.05,
        right=0.95,
        bottom=0.15,
        top=0.9,
        wspace=0.15,
        hspace=0.1,
    )

    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("3D Skeleton")

    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.set_xlabel("Probability (%)")
    ax_bar.set_title(f"Top-{topk} Predictions")

    ax_gap = fig.add_subplot(gs[1, 1], projection="3d")
    ax_gap.set_title("GAP Features (256→16×16)", fontsize=10)

    gap_grid = gap.reshape(16, 16)
    bar3d_colored(ax_gap, gap_grid, _GAP_CMAP)
    ax_gap.view_init(elev=gap_elev, azim=45)

    x_data = keypoint_clip[:, :, 0]
    y_data = keypoint_clip[:, :, 1]
    z_data = keypoint_clip[:, :, 2]

    valid_mask = ~np.all(np.all(np.isclose(keypoint_clip, 0), axis=2), axis=1)
    if np.any(valid_mask):
        valid_frames = keypoint_clip[valid_mask]
        all_points = valid_frames.reshape(-1, 3)
        center = np.mean(all_points, axis=0)
        max_extent = np.max(np.abs(all_points - center)) * 1.2
    else:
        center = np.zeros(3)
        max_extent = 1.0

    joint_plots = []
    for v in range(V):
        plot = ax3d.plot(
            [],
            [],
            [],
            marker="o",
            markersize=6,
            color=_NTU_JOINT_COLORS[v % len(_NTU_JOINT_COLORS)],
            linestyle="None",
            alpha=0.9,
        )[0]
        joint_plots.append(plot)

    bone_lines = []
    for v1, v2 in _NTU_BONE_PAIRS:
        if v1 < V and v2 < V:
            line = ax3d.plot([], [], [], color="gray", linewidth=2, alpha=0.7)[0]
            bone_lines.append((line, v1, v2))

    ax3d.set_xlim(center[0] - max_extent, center[0] + max_extent)
    ax3d.set_ylim(center[1] - max_extent, center[1] + max_extent)
    ax3d.set_zlim(center[2] - max_extent, center[2] + max_extent)

    y_pos = np.arange(topk)
    bar_height = 0.6
    cmap = plt.cm.Blues
    colors = []
    for i, idx in enumerate(topk_indices):
        if idx == true_label:
            colors.append("#2ecc71")
        else:
            shade = 0.9 - (i * 0.07)
            colors.append(cmap(shade))
    bars = ax_bar.barh(
        y_pos, topk_probs * 100, height=bar_height, color=colors, alpha=0.85
    )
    ax_bar.set_yticks(y_pos)
    class_labels = [label_name(task, int(idx)) for idx in topk_indices]
    ax_bar.set_yticklabels(class_labels, fontsize=8)
    ax_bar.set_xlim(0, 105)
    ax_bar.invert_yaxis()

    for i, (bar, idx) in enumerate(zip(bars, topk_indices)):
        prob_text = f"{topk_probs[i] * 100:.1f}%"
        ax_bar.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            prob_text,
            va="center",
            fontsize=8,
            fontweight="bold" if i == 0 else "normal",
        )
        if idx == true_label:
            ax_bar.text(
                0.5,
                bar.get_y() + bar.get_height() / 2,
                "TRUE",
                va="center",
                ha="left",
                fontsize=7,
                fontweight="bold",
                color="white",
            )

    frame_text = ax3d.text2D(0.02, 0.98, "", transform=ax3d.transAxes, fontsize=10)

    ui_state = {
        "frame": 0,
        "paused": False,
        "anim": None,
    }

    def update_frame(frame_idx):
        """Update visualization for given frame."""
        frame_data = keypoint_clip[frame_idx]

        for v in range(V):
            x, y, z = frame_data[v]
            joint_plots[v].set_data([x], [y])
            joint_plots[v].set_3d_properties([z])

        for line, v1, v2 in bone_lines:
            x1, y1, z1 = frame_data[v1]
            x2, y2, z2 = frame_data[v2]
            if not (np.allclose([x1, y1, z1], 0) and np.allclose([x2, y2, z2], 0)):
                line.set_data([x1, x2], [y1, y2])
                line.set_3d_properties([z1, z2])
            else:
                line.set_data([], [])
                line.set_3d_properties([])

        frame_text.set_text(f"Frame: {frame_idx + 1}/{num_frames}")

        return joint_plots + [l for l, _, _ in bone_lines] + [frame_text]

    ax_slider = plt.axes([0.2, 0.08, 0.55, 0.03])
    frame_slider = Slider(ax_slider, "Frame", 0, num_frames - 1, valinit=0, valfmt="%d")

    ax_button = plt.axes([0.8, 0.08, 0.1, 0.04])
    button = Button(ax_button, "Pause")

    def on_frame_change(val):
        """Handle slider changes."""
        frame_idx = int(val)
        ui_state["frame"] = frame_idx
        ui_state["paused"] = True
        button.label.set_text("Play")
        if ui_state["anim"]:
            ui_state["anim"].event_source.stop()
        update_frame(frame_idx)
        fig.canvas.draw_idle()

    frame_slider.on_changed(on_frame_change)

    def toggle_pause(event):
        """Toggle play/pause."""
        ui_state["paused"] = not ui_state["paused"]
        if ui_state["paused"]:
            button.label.set_text("Play")
            if ui_state["anim"]:
                ui_state["anim"].event_source.stop()
        else:
            button.label.set_text("Pause")
            if ui_state["anim"]:
                ui_state["anim"].event_source.start()

    button.on_clicked(toggle_pause)

    def animate(frame):
        """Animation callback."""
        if ui_state["paused"]:
            return joint_plots + [l for l, _, _ in bone_lines] + [frame_text]

        ui_state["frame"] = frame
        frame_slider.eventson = False
        frame_slider.set_val(frame)
        frame_slider.eventson = True
        ax_gap.view_init(elev=gap_elev, azim=45 + frame * gap_spin_deg_per_frame)

        return update_frame(frame)

    interval_ms = int(1000 / fps)
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=num_frames,
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    ui_state["anim"] = ani

    return fig, ani


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ST-GCN++ single clip inference on NTU RGB+D with visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default="data/ntu120_3danno.pkl",
        help="Path to the NTU 3-D skeleton pickle file",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the pre-trained .pth checkpoint",
    )
    parser.add_argument(
        "--modality",
        choices=["joint", "bone"],
        default="joint",
        help="Input modality: 'joint' (raw coordinates) or 'bone' (edge vectors)",
    )
    parser.add_argument(
        "--split",
        default="xsub_val",
        help="Dataset split (e.g. xsub_train, xsub_val, xset_train, xset_val)",
    )
    parser.add_argument(
        "--clip",
        type=int,
        default=0,
        help="Clip number within the split (0-indexed)",
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=10,
        help="Number of temporal clips per sample for test-time augmentation",
    )
    parser.add_argument(
        "--clip-len",
        type=int,
        default=100,
        help="Number of frames per clip",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Animation frames per second",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization, only print results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested via --device, but torch.cuda.is_available() is False "
            "in this environment."
        )

    device = torch.device(args.device)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("=" * 60)
    print("ST-GCN++ Single Clip Inference")
    print("=" * 60)
    print(f"Device    : {device}")
    print(f"Data      : {args.data}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split     : {args.split}")
    print(f"Modality  : {args.modality}")
    print(f"Clip #    : {args.clip} (from {args.split})")
    print(f"Clips     : {args.num_clips} × {args.clip_len} frames")
    print("=" * 60)

    print("\nLoading annotation...")
    annotation = load_annotation(args.data)

    sample_info = get_sample_info(annotation, args.split, args.clip)
    print(f"\nSample Info:")
    print(f"  Clip #      : {sample_info['index']}")
    print(f"  Frame Dir   : {sample_info['frame_dir']}")
    print(f"  True Label  : {sample_info['label']}")
    print(f"  Total Frames: {sample_info['total_frames']}")

    sample_info["age_years"] = resolve_age_years(
        sample_info["frame_dir"], Path("../data/Comp_v6_KLD01/train/ages")
    )
    if sample_info["age_years"] is not None:
        print(f"  Age (years) : {sample_info['age_years']:.1f}")

    print("\nBuilding dataloader...")
    dataloader = build_dataloader(
        pkl_path=args.data,
        split=args.split,
        modality=args.modality,
        clip_len=args.clip_len,
        num_clips=args.num_clips,
        batch_size=1,
        num_workers=0,
        max_samples=args.clip + 1,
    )
    print(f"  Dataset size: {len(dataloader.dataset)} samples")

    print("\nInspecting checkpoint...")
    _, state_dict = load_checkpoint_raw(str(ckpt_path), device)
    task = detect_task_from_state_dict(state_dict)
    print(f"  Detected task: {task}")

    print("\nBuilding model...")
    model = build_model_for_task(task)
    model.to(device)

    print(f"\nLoading checkpoint: {ckpt_path}")
    missing, unexpected, skipped = load_checkpoint(model, state_dict)
    if missing:
        print(
            f"  [WARN] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        print(
            f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
    if skipped:
        print(f"  [INFO] Skipped {len(skipped)} keys not belonging to backbone/head")

    print("\nRunning inference...")
    true_label, probs, keypoint, gap = inference_single(
        model, dataloader, device, args.clip
    )

    topk = min(5, probs.shape[1])
    topk_probs, topk_indices = probs[0].topk(topk)

    true_class_name = label_name(task, int(true_label))

    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"\nTask      : {task}")
    print(f'True Label: {true_label} - "{true_class_name}"')
    if sample_info.get("age_years") is not None:
        print(f"True Age  : {sample_info['age_years']:.1f} years")
    print("\nTop-5 Predictions:")
    for i, (prob, idx) in enumerate(zip(topk_probs, topk_indices), 1):
        class_name = label_name(task, idx.item())
        marker = " <-- TRUE LABEL" if idx.item() == true_label else ""
        print(f"  {i}. {class_name:<25} | {prob.item() * 100:>6.2f}%{marker}")
    print("=" * 60)

    if not args.no_viz:
        viz_joints = prepare_visualization_joints(
            sample_info["keypoint"],
            clip_len=args.clip_len,
            num_clips=args.num_clips,
        )
        NC, M, T, V, C = viz_joints.shape
        top_prediction = label_name(task, int(topk_indices[0].item()))
        print("\nLaunching visualization...")
        print(f"  Modality : {args.modality} (visualization always shows joints)")
        print(f"  Skeleton : {V} joints, {T} frames")
        print(f"  GAP      : {gap.shape[0]} features")
        print(
            f'  Top prediction: "{top_prediction}" ({topk_probs[0].item() * 100:.1f}%)'
        )
        print(f'  True label: "{true_class_name}"')
        if sample_info.get("age_years") is not None:
            print(f"  True age: {sample_info['age_years']:.1f} years")
        print("\nControls:")
        print("  - Drag slider to scrub through frames")
        print("  - Click Pause/Play to control animation")
        print("  - Close window to exit")
        fig, ani = create_visualization(
            keypoint=viz_joints,
            task=task,
            true_label=true_label,
            probs=probs.cpu().numpy(),
            sample_info=sample_info,
            gap=gap,
            fps=args.fps,
        )
        plt.show()


if __name__ == "__main__":
    main()
