#!/usr/bin/env python3
"""
Create joint angular-velocity heatmaps from HumanML3D NPZ files.

The HumanML3D NPZ files store joint positions (T, 22, 3) at 20 FPS.
We estimate joint angles from adjacent bones and take the first derivative
to get angular velocity in radians/second.

Usage examples:
  # Single trial heatmap
  python3 7_joint_velocity_heatmap.py --input data/humanml3d_joints_4/SUBJ01/SUBJ1_0_humanml3d_22joints.npz --output outputs/SUBJ01_trial0_heatmap.png

  # Batch over all trials in data/humanml3d_joints_4/
  python3 7_joint_velocity_heatmap.py --all --output-dir outputs/velocity_heatmaps

  # Normalize time to 100 points (0-100%)
  python3 7_joint_velocity_heatmap.py --input data/humanml3d_joints_4/SUBJ01/SUBJ1_0_humanml3d_22joints.npz --normalize 100
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import interp1d


JOINT_NAMES = [
    "pelvis",
    "L_hip",
    "R_hip",
    "spine1",
    "L_knee",
    "R_knee",
    "spine2",
    "L_ankle",
    "R_ankle",
    "spine3",
    "L_foot",
    "R_foot",
    "neck",
    "L_collar",
    "R_collar",
    "head",
    "L_shoulder",
    "R_shoulder",
    "L_elbow",
    "R_elbow",
    "L_wrist",
    "R_wrist",
]

KINEMATIC_CHAIN = [
    (0, 1),
    (1, 4),
    (4, 7),
    (7, 10),  # Left Leg
    (0, 2),
    (2, 5),
    (5, 8),
    (8, 11),  # Right Leg
    (0, 3),
    (3, 6),
    (6, 9),
    (9, 12),
    (12, 15),  # Spine
    (9, 13),
    (13, 16),
    (16, 18),
    (18, 20),  # Left Arm
    (9, 14),
    (14, 17),
    (17, 19),
    (19, 21),  # Right Arm
]

PREFERRED_CHILD = {
    0: 3,  # pelvis -> spine1
    3: 6,  # spine1 -> spine2
    6: 9,  # spine2 -> spine3
    9: 12,  # spine3 -> neck
    12: 15,  # neck -> head
    1: 4,  # L_hip -> L_knee
    4: 7,  # L_knee -> L_ankle
    7: 10,  # L_ankle -> L_foot
    2: 5,  # R_hip -> R_knee
    5: 8,  # R_knee -> R_ankle
    8: 11,  # R_ankle -> R_foot
    13: 16,  # L_collar -> L_shoulder
    16: 18,  # L_shoulder -> L_elbow
    18: 20,  # L_elbow -> L_wrist
    14: 17,  # R_collar -> R_shoulder
    17: 19,  # R_shoulder -> R_elbow
    19: 21,  # R_elbow -> R_wrist
}


def build_parent_children(joint_count):
    parents = [None] * joint_count
    children = [[] for _ in range(joint_count)]
    for parent, child in KINEMATIC_CHAIN:
        parents[child] = parent
        children[parent].append(child)
    return parents, children


def safe_angle_between(v1, v2, eps=1e-8):
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    valid = (n1 > eps) & (n2 > eps)
    v1n = np.zeros_like(v1)
    v2n = np.zeros_like(v2)
    v1n[valid] = v1[valid] / n1[valid][..., None]
    v2n[valid] = v2[valid] / n2[valid][..., None]
    dot = np.sum(v1n * v2n, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    angle = np.zeros_like(dot)
    angle[valid] = np.arccos(dot[valid])
    return angle


def compute_joint_angle_series(joints, parents, children):
    T, J, _ = joints.shape
    angles = np.full((T, J), np.nan, dtype=np.float32)

    for j in range(J):
        parent = parents[j]
        child = PREFERRED_CHILD.get(j)
        if parent is None or child is None:
            continue
        if parent >= J or child >= J:
            continue
        v_parent = joints[:, parent] - joints[:, j]
        v_child = joints[:, child] - joints[:, j]
        angles[:, j] = safe_angle_between(v_parent, v_child).astype(np.float32)

    return angles


def compute_orientation_change(joints, parents, children, fps):
    T, J, _ = joints.shape
    velocities = np.zeros((T - 1, J), dtype=np.float32)
    for j in range(J):
        parent = parents[j]
        child = PREFERRED_CHILD.get(j)
        if parent is not None:
            vec = joints[:, j] - joints[:, parent]
        elif child is not None:
            vec = joints[:, child] - joints[:, j]
        else:
            continue
        ang = safe_angle_between(vec[1:], vec[:-1])
        velocities[:, j] = ang * fps
    return velocities


def compute_joint_angular_velocity(joints, fps):
    parents, children = build_parent_children(joints.shape[1])
    angles = compute_joint_angle_series(joints, parents, children)

    angle_vel = np.abs(np.diff(angles, axis=0)) * fps
    angle_vel = np.where(np.isfinite(angle_vel), angle_vel, np.nan)

    orientation_vel = compute_orientation_change(joints, parents, children, fps)
    filled = np.where(np.isnan(angle_vel), orientation_vel, angle_vel)
    return filled


def normalize_time(data, num_points):
    if num_points is None or num_points <= 0:
        return data
    T = data.shape[0]
    if T < 2:
        return np.zeros((num_points, data.shape[1]), dtype=data.dtype)
    original_times = np.linspace(0.0, 1.0, T)
    new_times = np.linspace(0.0, 1.0, num_points)
    normalized = np.zeros((num_points, data.shape[1]), dtype=data.dtype)
    for j in range(data.shape[1]):
        interp_func = interp1d(
            original_times,
            data[:, j],
            kind="linear",
            fill_value="extrapolate",  # type: ignore[arg-type]
        )
        normalized[:, j] = interp_func(new_times)
    return normalized


def build_velocity_colormap(vmax):
    stop_blue = 4.0 / max(vmax, 1e-6)
    stop_blue = min(max(stop_blue, 0.0), 1.0)
    stops = [
        (0.0, (1.0, 1.0, 1.0)),
        (stop_blue, (0.0, 0.35, 0.95)),
        ((1.0 + stop_blue) / 2.0, (1.0, 0.9, 0.0)),
        (1.0, (0.9, 0.0, 0.0)),
    ]
    return LinearSegmentedColormap.from_list("velocity_custom", stops)


def map_velocity_to_rgba(values, vmax):
    blue = np.array([0.0, 0.35, 0.95])
    yellow = np.array([1.0, 0.9, 0.0])
    red = np.array([0.9, 0.0, 0.0])

    rgba = np.ones(values.shape + (4,), dtype=np.float32)
    rgba[..., :3] = 1.0
    rgba[..., 3] = 0.0

    finite_mask = np.isfinite(values)
    v = np.where(finite_mask, values, 0.0)

    low_mask = (v <= 4.0) & finite_mask
    rgba[low_mask, :3] = blue
    rgba[low_mask, 3] = np.clip(v[low_mask] / 4.0, 0.0, 1.0)

    high_mask = (v > 4.0) & finite_mask
    if np.any(high_mask):
        vmax_adj = max(vmax, 4.0 + 1e-6)
        t = (v[high_mask] - 4.0) / (vmax_adj - 4.0)
        t = np.clip(t, 0.0, 1.0)
        mid_mask = t <= 0.5
        t1 = np.where(mid_mask, t / 0.5, 0.0)
        t2 = np.where(~mid_mask, (t - 0.5) / 0.5, 0.0)
        color = np.zeros((t.shape[0], 3), dtype=np.float32)
        if np.any(mid_mask):
            color[mid_mask] = blue + (yellow - blue) * t1[mid_mask, None]
        if np.any(~mid_mask):
            color[~mid_mask] = yellow + (red - yellow) * t2[~mid_mask, None]
        rgba[high_mask, :3] = color
        rgba[high_mask, 3] = 1.0

    return rgba


def plot_heatmap(velocities, fps, title, normalize_points=None, output_path=None):
    if normalize_points:
        vel_plot = normalize_time(velocities, normalize_points)
        x_label = "Normalized Gait Cycle (%)"
        x_ticks = np.linspace(0, normalize_points - 1, 11)
        x_ticklabels = [f"{int(x)}%" for x in np.linspace(0, 100, 11)]
    else:
        vel_plot = velocities
        x_label = "Time (s)"
        max_t = vel_plot.shape[0] / max(fps, 1e-6)
        x_ticks = np.linspace(0, vel_plot.shape[0] - 1, 6)
        x_ticklabels = [f"{(x / fps):.2f}" for x in x_ticks]

    vmax = np.nanpercentile(vel_plot, 99) if np.any(np.isfinite(vel_plot)) else 1.0
    vmax = max(vmax, 4.0)
    rgba = map_velocity_to_rgba(vel_plot, vmax)
    cmap = build_velocity_colormap(vmax)
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(
        np.transpose(rgba, (1, 0, 2)),
        aspect="auto",
        origin="lower",
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("Joint", fontsize=11)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_yticks(range(len(JOINT_NAMES)))
    ax.set_yticklabels(JOINT_NAMES, fontsize=9)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=9)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Angular Velocity (rad/s)", fontsize=10)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
    return fig


def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    if "joints" not in data:
        raise ValueError(f"'joints' array not found in {npz_path}")
    joints = data["joints"]
    fps = float(data.get("fps", 20))
    subject_id = str(data.get("subject_id", ""))
    trial_name = str(data.get("trial_name", Path(npz_path).stem))
    age = data.get("age", None)
    sex = data.get("sex", None)
    return joints, fps, subject_id, trial_name, age, sex


def collect_npz_files(data_dir):
    data_path = Path(data_dir)
    return sorted(data_path.rglob("*.npz"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create joint angular velocity heatmaps from HumanML3D NPZ files"
    )
    parser.add_argument("--input", type=str, help="Path to a single HumanML3D NPZ file")
    parser.add_argument("--output", type=str, help="Output image path (PNG)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/humanml3d_joints_4",
        help="HumanML3D joints directory",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for batch mode"
    )
    parser.add_argument(
        "--all", action="store_true", help="Process all NPZ files under data-dir"
    )
    parser.add_argument(
        "--normalize",
        type=int,
        default=0,
        help="Normalize time to N points (0 = no normalization)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all:
        output_dir = Path(args.output_dir) if args.output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        npz_files = collect_npz_files(args.data_dir)
        if not npz_files:
            raise FileNotFoundError(f"No NPZ files found in {args.data_dir}")
        for npz_path in npz_files:
            joints, fps, subject_id, trial_name, age, sex = load_npz(npz_path)
            velocities = compute_joint_angular_velocity(joints, fps)
            title_parts = [subject_id or npz_path.parent.name, trial_name]
            if age is not None:
                title_parts.append(f"Age {age}")
            if sex is not None:
                title_parts.append(f"Sex {sex}")
            title = " | ".join([p for p in title_parts if p])
            out_path = None
            if output_dir:
                out_name = f"{Path(npz_path).stem}_joint_angvel_heatmap.png"
                out_path = output_dir / out_name
            plot_heatmap(
                velocities,
                fps,
                title,
                normalize_points=args.normalize,
                output_path=out_path,
            )
            plt.close("all")
        return

    if not args.input:
        raise ValueError("--input is required unless --all is used")

    npz_path = Path(args.input)
    joints, fps, subject_id, trial_name, age, sex = load_npz(npz_path)
    velocities = compute_joint_angular_velocity(joints, fps)

    title_parts = [subject_id or npz_path.parent.name, trial_name]
    if age is not None:
        title_parts.append(f"Age {age}")
    if sex is not None:
        title_parts.append(f"Sex {sex}")
    title = " | ".join([p for p in title_parts if p])

    fig = plot_heatmap(
        velocities, fps, title, normalize_points=args.normalize, output_path=args.output
    )
    if not args.output:
        plt.show()


if __name__ == "__main__":
    main()
