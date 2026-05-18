#!/usr/bin/env python3
"""
Gait Analysis for LoRA-MDM Age Dataset
Computes spatiotemporal parameters, gait variability (CV), and joint ROM
from the ground-truth dataset (humanml3d_new_joints_6) and LoRA-generated
motion clips (generated_clips_humanml3d/motion1_walk_forward).

Outputs all figures to outputs/gait_analysis/{dataset,generated,combined}/
"""

import os
import sys
import glob
import warnings
import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.signal import find_peaks
from scipy.stats import linregress

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA = ROOT / "data"
JOINTS_DIR = DATA / "humanml3d_new_joints_6"
GEN_DIR = DATA / "generated_clips_humanml3d"
OUT_DIR = ROOT / "outputs" / "gait_analysis"
for sub in ("dataset", "generated", "combined"):
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
FPS = 20
DT = 1.0 / FPS
N_GAIT_PTS = 101  # normalised gait-cycle samples (0 % … 100 %)
MIN_CLIP_FRAMES = 20  # ~1 s; skip shorter clips

# ── HumanML3D t2m 22-joint indices ────────────────────────────────────────────
# kinematic chains: [0,2,5,8,11], [0,1,4,7,10], [0,3,6,9,12,15],
#                  [9,14,17,19,21], [9,13,16,18,20]
PELVIS = 0
L_HIP, R_HIP = 1, 2
SPINE1 = 3
L_KNEE, R_KNEE = 4, 5
SPINE2 = 6
L_ANKLE, R_ANKLE = 7, 8
SPINE3 = 9
L_FOOT, R_FOOT = 10, 11  # toe joints
NECK = 12
L_COLLAR, R_COLLAR = 13, 14
HEAD = 15
L_SHOULDER, R_SHOULDER = 16, 17
L_ELBOW, R_ELBOW = 18, 19
L_WRIST, R_WRIST = 20, 21

JOINT_NAMES = [
    "pelvis", "L_hip", "R_hip", "spine1",
    "L_knee", "R_knee", "spine2", "L_ankle",
    "R_ankle", "spine3", "L_foot", "R_foot",
    "neck", "L_collar", "R_collar", "head",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist",
]

# Heatmap display order: upper body (top) → lower body (bottom)
HEATMAP_ORDER = [
    R_WRIST, L_WRIST, R_ELBOW, L_ELBOW,
    R_SHOULDER, L_SHOULDER, HEAD, R_COLLAR, L_COLLAR, NECK,
    R_FOOT, L_FOOT, SPINE3, R_ANKLE, L_ANKLE, SPINE2,
    R_KNEE, L_KNEE, SPINE1, R_HIP, L_HIP, PELVIS,
]
HEATMAP_LABELS = [JOINT_NAMES[i] for i in HEATMAP_ORDER]

GENERATED_GROUPS = ["young", "mid", "old"]
GROUP_COLORS = {"young": "#2196F3", "mid": "#FF9800", "old": "#F44336"}
AGE_BINS = {"young": (21, 40), "mid": (40, 65), "old": (65, 100)}

# ── Metadata ───────────────────────────────────────────────────────────────────

def load_metadata():
    spec = importlib.util.spec_from_file_location(
        "metadata", DATA / "van_criekinge_unprocessed_1" / "metadata.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {**mod.create_able_bodied_metadata(), **mod.create_stroke_metadata()}

# ── Gait analysis primitives ───────────────────────────────────────────────────

def get_forward_axis(pelvis_pos):
    """Return 0 (X) or 2 (Z) – whichever shows more displacement."""
    xz = np.array([np.ptp(pelvis_pos[:, 0]), np.ptp(pelvis_pos[:, 2])])
    return 0 if xz[0] > xz[1] else 2


def smooth_1d(y, wl=5):
    """Simple moving-average smoothing (handles short arrays)."""
    wl = min(wl, len(y))
    if wl < 3:
        return y.copy()
    kernel = np.ones(wl) / wl
    pad = wl // 2
    padded = np.pad(y, pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(y)]


# Contact-detection thresholds (tuned for HumanML3D metre-scale data)
_HEIGHT_THRESH = 0.05   # max relative height above per-clip floor (m)
_VEL_THRESH    = 1.0    # max foot speed to count as "in stance" (m/s)
_MIN_CONTACT   = 3      # minimum consecutive frames to call it a stance event


def detect_foot_contacts(joints, foot_idx):
    """
    Robust foot-contact detection using **relative height + velocity**.

    Instead of comparing against absolute Y=0 (which varies per clip and
    per subject), we:
      1. Estimate the per-clip floor as the 2nd-percentile of foot Y.
      2. Flag frames where (foot_Y - floor) < _HEIGHT_THRESH  AND
         foot speed < _VEL_THRESH.
      3. Group consecutive flagged frames into stance events; the first
         frame of each event is the heel-strike.

    Returns an int array of heel-strike frame indices.
    """
    foot_pos = joints[:, foot_idx]          # (T, 3)
    foot_y   = foot_pos[:, 1]

    # Per-clip floor: 2nd percentile is robust to a few outlier low frames
    floor_y     = np.percentile(foot_y, 2)
    rel_height  = foot_y - floor_y

    # Foot speed: finite-difference of 3-D position, forward-padded
    vel   = np.diff(foot_pos, axis=0, prepend=foot_pos[:1]) * FPS
    speed = np.linalg.norm(vel, axis=-1)

    in_contact = (rel_height < _HEIGHT_THRESH) & (speed < _VEL_THRESH)

    # Find rising edges (False→True transitions) = heel strikes
    padded  = np.concatenate([[False], in_contact, [False]])
    starts  = np.where(~padded[:-1] &  padded[1:])[0]
    ends    = np.where( padded[:-1] & ~padded[1:])[0]

    # Keep only stance events long enough to be real
    heel_strikes = [s for s, e in zip(starts, ends) if (e - s) >= _MIN_CONTACT]
    return np.array(heel_strikes, dtype=int)


def compute_gait_metrics(joints):
    """
    Compute walking speed, stride length, cadence, and stride CV from
    a (T, 22, 3) joint array.  Returns a dict or None if data is thin.

    Walking speed is computed directly from the pelvis trajectory (robust
    to imperfect stride detection).  CV is only computed when ≥3 strides
    are available per side so it isn't dominated by 1-sample noise.
    """
    T = joints.shape[0]
    if T < MIN_CLIP_FRAMES:
        return None

    fwd = get_forward_axis(joints[:, PELVIS])

    l_strikes = detect_foot_contacts(joints, L_FOOT)
    r_strikes = detect_foot_contacts(joints, R_FOOT)

    if len(l_strikes) < 2 and len(r_strikes) < 2:
        return None

    stride_times, stride_lengths = [], []
    for strikes in (l_strikes, r_strikes):
        for i in range(len(strikes) - 1):
            t0, t1 = strikes[i], strikes[i + 1]
            dt = (t1 - t0) * DT
            # Stride length = pelvis displacement over one stride
            dl = abs(joints[t1, PELVIS, fwd] - joints[t0, PELVIS, fwd])
            # Sanity bounds: 0.4–3 s stride time, 0.1–3 m stride length
            if 0.4 <= dt <= 3.0 and 0.1 <= dl <= 3.0:
                stride_times.append(dt)
                stride_lengths.append(dl)

    if not stride_times:
        return None

    st = np.array(stride_times)
    sl = np.array(stride_lengths)

    # Speed from full pelvis arc (independent of stride segmentation)
    total_dist = abs(joints[-1, PELVIS, fwd] - joints[0, PELVIS, fwd])
    speed      = total_dist / (T * DT) if T > 0 else 0.0
    cadence    = 60.0 / float(np.mean(st))

    # CV only meaningful with ≥3 strides; single-stride clips → NaN
    cv_time   = float(np.std(st) / np.mean(st) * 100) if len(st) >= 3 else np.nan
    cv_length = float(np.std(sl) / np.mean(sl) * 100) if len(sl) >= 3 else np.nan

    return {
        "speed":         float(speed),
        "stride_length": float(np.mean(sl)),
        "stride_time":   float(np.mean(st)),
        "cadence":       float(cadence),
        "cv_time":       cv_time,
        "cv_length":     cv_length,
        "n_strides":     len(st),
    }


def joint_angle_deg(p1, p2, p3):
    """
    Included angle (degrees) at vertex p2, between rays p1→p2 and p3→p2.
    Inputs: (..., 3).
    """
    v1 = p1 - p2
    v2 = p3 - p2
    n1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    cos_a = np.sum(v1 * v2, axis=-1) / (n1[..., 0] * n2[..., 0] + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def compute_rom(joints):
    """
    Hip and knee range of motion (degrees) from a (T, 22, 3) clip.
    """
    if joints.shape[0] < 5:
        return None
    l_knee_ang = joint_angle_deg(joints[:, L_HIP], joints[:, L_KNEE], joints[:, L_ANKLE])
    r_knee_ang = joint_angle_deg(joints[:, R_HIP], joints[:, R_KNEE], joints[:, R_ANKLE])
    l_hip_ang  = joint_angle_deg(joints[:, SPINE1], joints[:, L_HIP], joints[:, L_KNEE])
    r_hip_ang  = joint_angle_deg(joints[:, SPINE1], joints[:, R_HIP], joints[:, R_KNEE])
    return {
        "knee_rom": float(np.ptp(np.concatenate([l_knee_ang, r_knee_ang]))),
        "hip_rom":  float(np.ptp(np.concatenate([l_hip_ang,  r_hip_ang]))),
        "knee_mean": float(np.mean(np.concatenate([l_knee_ang, r_knee_ang]))),
        "hip_mean":  float(np.mean(np.concatenate([l_hip_ang,  r_hip_ang]))),
    }


def normalise_gait_cycle(joints, contacts):
    """
    Resample each stride (contact[i] → contact[i+1]) to N_GAIT_PTS frames.
    Returns (n_cycles, N_GAIT_PTS, 22, 3) or None.
    """
    if len(contacts) < 2:
        return None
    cycles = []
    src_t = np.linspace(0.0, 1.0, 1)  # placeholder
    tgt_t = np.linspace(0.0, 1.0, N_GAIT_PTS)
    for i in range(len(contacts) - 1):
        t0, t1 = int(contacts[i]), int(contacts[i + 1])
        seg = joints[t0 : t1 + 1]  # (seg_len, 22, 3)
        if len(seg) < 4:
            continue
        src_t = np.linspace(0.0, 1.0, len(seg))
        out = np.stack(
            [np.interp(tgt_t, src_t, seg[:, j, ax])
             for j in range(22) for ax in range(3)]
        ).reshape(22, 3, N_GAIT_PTS).transpose(2, 0, 1)  # (N_GAIT_PTS, 22, 3)
        cycles.append(out)
    return np.array(cycles) if cycles else None


def velocity_map_from_cycles(joints, contacts):
    """
    Mean joint speed (m/s) over the normalised gait cycle.
    Returns (N_GAIT_PTS, 22) or None.
    """
    cycles = normalise_gait_cycle(joints, contacts)
    if cycles is None or len(cycles) == 0:
        return None
    vel = np.diff(cycles, axis=1) * FPS        # (n, N-1, 22, 3)
    speed = np.linalg.norm(vel, axis=-1)       # (n, N-1, 22)
    speed = np.concatenate([speed, speed[:, -1:]], axis=1)  # pad → (n, N, 22)
    return np.mean(speed, axis=0)              # (N, 22)


def aggregate_metrics(metric_list, keys):
    """Return {key: mean, key_std: std} across list of dicts."""
    out = {}
    for k in keys:
        vals = [m[k] for m in metric_list if k in m and not np.isnan(m[k])]
        out[k] = float(np.mean(vals)) if vals else np.nan
        out[f"{k}_std"] = float(np.std(vals)) if len(vals) > 1 else np.nan
    return out

# ── Dataset analysis ───────────────────────────────────────────────────────────

def analyze_dataset(metadata):
    files = sorted(glob.glob(str(JOINTS_DIR / "*.npy")))
    subj_files = defaultdict(list)
    for f in files:
        sid = Path(f).stem.split("_")[0]
        subj_files[sid].append(f)

    results = []
    for sid, flist in sorted(subj_files.items()):
        if sid not in metadata:
            continue
        meta = metadata[sid]
        age = meta.get("age")
        if age is None:
            continue

        all_gait, all_rom, vmaps = [], [], []
        for f in flist:
            joints = np.load(f)          # (T, 22, 3)
            gm = compute_gait_metrics(joints)
            if gm and gm["n_strides"] >= 1:
                all_gait.append(gm)
            rm = compute_rom(joints)
            if rm:
                all_rom.append(rm)
            contacts = detect_foot_contacts(joints, L_FOOT)
            if len(contacts) >= 2:
                vm = velocity_map_from_cycles(joints, contacts)
                if vm is not None:
                    vmaps.append(vm)

        if not all_gait and not all_rom:
            continue

        rec = {
            "subject_id": sid,
            "age": age,
            "sex": meta.get("sex", "?"),
            "condition": meta.get("condition", "unknown"),
            "leg_length": meta.get("leg_length_m", np.nan),
        }
        if all_gait:
            gait_keys = ["speed", "stride_length", "stride_time", "cadence", "cv_time", "cv_length"]
            rec.update(aggregate_metrics(all_gait, gait_keys))
        if all_rom:
            rec.update(aggregate_metrics(all_rom, ["knee_rom", "hip_rom", "knee_mean", "hip_mean"]))
        if vmaps:
            rec["velocity_map"] = np.mean(vmaps, axis=0)  # (N_GAIT_PTS, 22)

        results.append(rec)

    print(f"  {len(results)} subjects analysed")
    return results

# ── Generated clips analysis ───────────────────────────────────────────────────

def load_generated(motion_dir, age_group, max_batches=16):
    """Return list of (T, 22, 3) arrays for one motion/age_group."""
    batch_files = sorted(glob.glob(str(motion_dir / age_group / "batch_*.npy")))[:max_batches]
    clips = []
    for bf in batch_files:
        data = np.load(bf, allow_pickle=True).item()
        motion = data["motion"]    # (16, 22, 3, 120)
        lengths = data["lengths"]  # (16,)
        for i in range(motion.shape[0]):
            clip = motion[i].transpose(2, 0, 1)[: int(lengths[i])]  # (T, 22, 3)
            clips.append(clip)
    return clips


def analyze_generated(motion_dir, age_group, max_batches=16):
    clips = load_generated(motion_dir, age_group, max_batches)
    all_combined, vmaps = [], []
    for joints in clips:
        gm = compute_gait_metrics(joints)
        rm = compute_rom(joints)
        # Merge gait + ROM into one per-clip dict so the violin plot can
        # access any metric from a single list.
        if gm and gm["n_strides"] >= 1:
            entry = dict(gm)
            if rm:
                entry.update(rm)
            all_combined.append(entry)
        contacts = detect_foot_contacts(joints, L_FOOT)
        if len(contacts) >= 2:
            vm = velocity_map_from_cycles(joints, contacts)
            if vm is not None:
                vmaps.append(vm)

    gait_keys = ["speed", "stride_length", "stride_time", "cadence", "cv_time", "cv_length"]
    rom_keys = ["knee_rom", "hip_rom"]
    rec = {"age_group": age_group, "n_clips": len(clips)}
    if all_combined:
        rec.update(aggregate_metrics(all_combined, gait_keys + rom_keys))
    if vmaps:
        rec["velocity_map"] = np.mean(vmaps, axis=0)

    return rec, all_combined

# ── Plotting helpers ───────────────────────────────────────────────────────────

def _ax_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)


def _regression_on_ax(ax, ages, vals, color="#1565C0"):
    ages = np.asarray(ages, float)
    vals = np.asarray(vals, float)
    mask = ~np.isnan(vals)
    ax_ages, ax_vals = ages[mask], vals[mask]
    if len(ax_ages) < 5:
        return
    ax.scatter(ax_ages, ax_vals, alpha=0.45, s=22, color=color, zorder=2)
    slope, intercept, r, p, _ = linregress(ax_ages, ax_vals)
    xl = np.array([ax_ages.min(), ax_ages.max()])
    ax.plot(xl, slope * xl + intercept, color=color, lw=2, zorder=3)
    star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
    ax.text(0.97, 0.06, f"r={r:.2f}{star}", transform=ax.transAxes,
            ha="right", fontsize=8, color=color)


def heatmap_grid(vmaps_by_label, title, out_path, shared_vmax=True):
    """Generic heatmap grid (reference-image style)."""
    n = len(vmaps_by_label)
    if n == 0:
        return
    vmax = max(float(np.percentile(vm, 98)) for _, vm in vmaps_by_label if vm is not None)
    fig, axes = plt.subplots(1, n, figsize=(11 * n, 8), sharey=True)
    if n == 1:
        axes = [axes]
    im = None
    for ax, (label, vm) in zip(axes, vmaps_by_label):
        if vm is None:
            ax.set_visible(False)
            continue
        data = vm[:, HEATMAP_ORDER].T  # (22, N_GAIT_PTS)
        im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r",
                       extent=[0, 100, -0.5, 21.5], origin="lower",
                       vmin=0, vmax=vmax if shared_vmax else None)
        ax.set_yticks(range(22))
        ax.set_yticklabels(HEATMAP_LABELS, fontsize=7.5)
        ax.set_xticks(range(0, 101, 10))
        ax.set_xticklabels([f"{x}%" for x in range(0, 101, 10)], fontsize=7.5, rotation=45)
        ax.set_xlabel("Normalized Gait Cycle (%)", fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.55, label="Joint Speed (m/s)")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")

# ── Plot 1: scatter + regression grid (dataset) ────────────────────────────────

def plot_scatter_grid(dataset_results, out_path):
    METRICS = [
        ("speed",         "Walking Speed (m/s)"),
        ("stride_length", "Stride Length (m)"),
        ("cadence",       "Cadence (strides/min)"),
        ("stride_time",   "Stride Time (s)"),
        ("cv_time",       "Stride Time CV (%)"),
        ("cv_length",     "Stride Length CV (%)"),
        ("knee_rom",      "Knee ROM (°)"),
        ("hip_rom",       "Hip ROM (°)"),
    ]
    ages = [r["age"] for r in dataset_results]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (key, label) in zip(axes.flatten(), METRICS):
        vals = [r.get(key, np.nan) for r in dataset_results]
        _regression_on_ax(ax, ages, vals)
        ax.set_xlabel("Age (years)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        _ax_style(ax)
    fig.suptitle("Gait Metrics vs Age — Ground-Truth Dataset (n=138)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")

# ── Plot 2: per-subject heatmaps (reference-image style) ──────────────────────

def plot_individual_heatmaps(dataset_results, out_dir):
    old_candidates = sorted(
        [r for r in dataset_results if r["age"] >= 70 and "velocity_map" in r],
        key=lambda r: -r["age"],
    )
    young_candidates = sorted(
        [r for r in dataset_results if r["age"] <= 35 and "velocity_map" in r],
        key=lambda r: r["age"],
    )
    if not old_candidates or not young_candidates:
        print("  Skipping individual heatmaps (missing candidates)")
        return
    old = old_candidates[0]
    young = young_candidates[0]
    heatmap_grid(
        [
            (f"Subject Age {old['age']}, Speed Normalized",   old["velocity_map"]),
            (f"Subject Age {young['age']}, Speed Normalized", young["velocity_map"]),
        ],
        "Joint Speed over Gait Cycle — Old vs Young",
        out_dir / "heatmap_old_vs_young.png",
    )

# ── Plot 3: mean heatmap per dataset age group ────────────────────────────────

def plot_age_group_heatmaps(dataset_results, out_dir):
    pairs = []
    for gname, (lo, hi) in AGE_BINS.items():
        vmaps = [r["velocity_map"] for r in dataset_results
                 if lo <= r["age"] < hi and "velocity_map" in r]
        if vmaps:
            pairs.append((f"{gname.capitalize()} ({lo}–{hi} y, n={len(vmaps)})",
                          np.mean(vmaps, axis=0)))
        else:
            pairs.append((f"{gname.capitalize()}", None))
    heatmap_grid(pairs,
                 "Mean Joint Speed Heatmap by Age Group — Dataset",
                 out_dir / "heatmap_dataset_by_age_group.png")

# ── Plot 4: correlation matrix ────────────────────────────────────────────────

def plot_correlation_matrix(dataset_results, out_path):
    KEYS = ["age", "speed", "stride_length", "cadence",
            "cv_time", "cv_length", "knee_rom", "hip_rom"]
    LBLS = ["Age", "Speed", "Stride\nLen", "Cadence",
            "CV\nTime", "CV\nLen", "Knee\nROM", "Hip\nROM"]
    n = len(KEYS)
    data = np.full((len(dataset_results), n), np.nan)
    for i, r in enumerate(dataset_results):
        for j, k in enumerate(KEYS):
            data[i, j] = r.get(k, np.nan)
    corr = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            mask = ~(np.isnan(data[:, i]) | np.isnan(data[:, j]))
            if mask.sum() >= 5:
                corr[i, j] = float(np.corrcoef(data[mask, i], data[mask, j])[0, 1])
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, mask=np.isnan(corr), annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1,
                xticklabels=LBLS, yticklabels=LBLS,
                ax=ax, square=True, linewidths=0.5,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Gait Metric Correlation Matrix — Dataset", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")

# ── Plot 5: violin plots by age group (dataset) ───────────────────────────────

def plot_violin_age_group(dataset_results, out_path):
    METRICS = [
        ("speed",         "Walking Speed (m/s)"),
        ("stride_length", "Stride Length (m)"),
        ("cv_time",       "Stride Time CV (%)"),
        ("knee_rom",      "Knee ROM (°)"),
        ("hip_rom",       "Hip ROM (°)"),
    ]
    color_list = ["#42A5F5", "#FFA726", "#EF5350"]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 5))
    for ax, (key, ylabel) in zip(axes, METRICS):
        groups_data = []
        labels = []
        for (gname, (lo, hi)), col in zip(AGE_BINS.items(), color_list):
            vals = [r.get(key, np.nan) for r in dataset_results
                    if lo <= r["age"] < hi and not np.isnan(r.get(key, np.nan))]
            if vals:
                groups_data.append(vals)
                labels.append(f"{gname.capitalize()}\n(n={len(vals)})")
        if groups_data:
            parts = ax.violinplot(groups_data, positions=range(len(groups_data)),
                                  showmedians=True)
            for pc, col in zip(parts["bodies"], color_list):
                pc.set_facecolor(col)
                pc.set_alpha(0.7)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        _ax_style(ax)
    fig.suptitle("Gait Metric Distribution by Age Group — Dataset",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")

# ── Plot 6: speed × stride-length coloured by age ────────────────────────────

def plot_speed_vs_stride(dataset_results, out_path):
    speeds = np.array([r.get("speed", np.nan) for r in dataset_results])
    strides = np.array([r.get("stride_length", np.nan) for r in dataset_results])
    ages = np.array([r["age"] for r in dataset_results], float)
    mask = ~(np.isnan(speeds) | np.isnan(strides))
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(speeds[mask], strides[mask], c=ages[mask],
                    cmap="RdYlGn_r", s=45, alpha=0.75, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Age (years)")
    ax.set_xlabel("Walking Speed (m/s)", fontsize=11)
    ax.set_ylabel("Stride Length (m)", fontsize=11)
    ax.set_title("Walking Speed vs Stride Length (coloured by Age)", fontsize=12, fontweight="bold")
    _ax_style(ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")

# ── Plot 7: box plots coloured by sex ────────────────────────────────────────

def plot_sex_age_group_boxes(dataset_results, out_path):
    """Box plots for walking speed, CV, ROM split by sex × age group."""
    METRICS = [
        ("speed",    "Walking Speed (m/s)"),
        ("cv_time",  "Stride Time CV (%)"),
        ("knee_rom", "Knee ROM (°)"),
    ]
    sex_colors = {"M": "#1565C0", "F": "#AD1457", "?": "#37474F"}
    fig, axes = plt.subplots(1, len(METRICS), figsize=(15, 5))
    group_names = list(AGE_BINS.keys())
    x = np.arange(len(group_names))

    for ax, (key, ylabel) in zip(axes, METRICS):
        for sex, color in sex_colors.items():
            offsets = {"M": -0.2, "F": 0.2, "?": 0.0}
            all_data = []
            for gname, (lo, hi) in AGE_BINS.items():
                vals = [r.get(key, np.nan) for r in dataset_results
                        if r["sex"] == sex and lo <= r["age"] < hi
                        and not np.isnan(r.get(key, np.nan))]
                all_data.append(vals)
            pos = x + offsets.get(sex, 0.0)
            bps = ax.boxplot(
                [d if d else [np.nan] for d in all_data],
                positions=pos, widths=0.3, patch_artist=True,
                medianprops=dict(color="white", linewidth=2),
                boxprops=dict(facecolor=color, alpha=0.6),
                flierprops=dict(marker=".", markersize=3, color=color, alpha=0.5),
                whiskerprops=dict(color=color), capprops=dict(color=color),
            )
            # legend proxy
            ax.scatter([], [], color=color, alpha=0.7,
                       label=f"{'Male' if sex=='M' else ('Female' if sex=='F' else 'Unknown')}")
        ax.set_xticks(x)
        ax.set_xticklabels([g.capitalize() for g in group_names], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        _ax_style(ax)

    fig.suptitle("Gait Metrics by Sex and Age Group — Dataset", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")

# ── Plot 8: generated clips – metric bars ─────────────────────────────────────

def plot_generated_bars(gen_by_group, out_path):
    METRICS = [
        ("speed",         "speed_std",         "Walking Speed (m/s)"),
        ("stride_length", "stride_length_std",  "Stride Length (m)"),
        ("cadence",       "cadence_std",        "Cadence (strides/min)"),
        ("stride_time",   "stride_time_std",    "Stride Time (s)"),
        ("cv_time",       "cv_time_std",        "Stride Time CV (%)"),
        ("cv_length",     "cv_length_std",      "Stride Length CV (%)"),
        ("knee_rom",      "knee_rom_std",       "Knee ROM (°)"),
        ("hip_rom",       "hip_rom_std",        "Hip ROM (°)"),
    ]
    groups = GENERATED_GROUPS
    colors = [GROUP_COLORS[g] for g in groups]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    for ax, (mkey, skey, ylabel) in zip(axes.flatten(), METRICS):
        means = [gen_by_group[g].get(mkey, np.nan) for g in groups]
        stds  = [gen_by_group[g].get(skey, np.nan)  for g in groups]
        x = np.arange(len(groups))
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([g.capitalize() for g in groups], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        _ax_style(ax)
    fig.suptitle("Gait Metrics by Age Group — LoRA-Generated Motion (walk forward)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")

# ── Plot 9: clinical threshold comparison ─────────────────────────────────────

def plot_clinical_thresholds(gen_by_group, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    checks = [
        {
            "title": "Walking Speed (m/s)",
            "vals": [
                ("Young\nGenerated", gen_by_group["young"].get("speed", np.nan), "#2196F3"),
                ("Old\nGenerated",   gen_by_group["old"].get("speed",   np.nan), "#F44336"),
                ("Young\nClinical",  1.35,  "#90CAF9"),
                ("Old\nClinical",    0.80,  "#EF9A9A"),
            ],
        },
        {
            "title": "Stride Time CV (%)",
            "vals": [
                ("Young\nGenerated", gen_by_group["young"].get("cv_time", np.nan), "#2196F3"),
                ("Old\nGenerated",   gen_by_group["old"].get("cv_time",   np.nan), "#F44336"),
                ("Young\nClinical",  2.0, "#90CAF9"),
                ("Old\nClinical",    3.5, "#EF9A9A"),
            ],
        },
        {
            "title": "Stride Length CV (%)",
            "vals": [
                ("Young\nGenerated", gen_by_group["young"].get("cv_length", np.nan), "#2196F3"),
                ("Old\nGenerated",   gen_by_group["old"].get("cv_length",   np.nan), "#F44336"),
                ("Young\nClinical",  2.0, "#90CAF9"),
                ("Old\nClinical",    2.7, "#EF9A9A"),
            ],
        },
    ]
    for ax, cfg in zip(axes, checks):
        labels, vals, colors = zip(*cfg["vals"])
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.55)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01 * max(v for v in vals if not np.isnan(v)),
                        f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(cfg["title"], fontsize=10, fontweight="bold")
        _ax_style(ax)
    # Legend for generated vs clinical
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#2196F3", label="Generated"),
               Patch(facecolor="#90CAF9", label="Clinical ref")]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle("Generated Gait Metrics vs Clinical Reference Thresholds",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")

# ── Plot 10: generated heatmaps ────────────────────────────────────────────────

def plot_generated_heatmaps(gen_by_group, out_path):
    pairs = [(f"{g.capitalize()} Age Group", gen_by_group[g].get("velocity_map"))
             for g in GENERATED_GROUPS]
    heatmap_grid(pairs,
                 "Joint Speed Heatmap by Age Group — LoRA-Generated Motion",
                 out_path)

# ── Plot 11: dataset vs generated side-by-side ───────────────────────────────

def plot_dataset_vs_generated(dataset_results, gen_by_group, out_path):
    METRICS = [
        ("speed",         "Walking Speed (m/s)"),
        ("stride_length", "Stride Length (m)"),
        ("cv_time",       "Stride Time CV (%)"),
        ("hip_rom",       "Hip ROM (°)"),
    ]
    group_labels = ["Young", "Mid", "Old"]
    x = np.arange(3)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 5))

    for ax, (key, ylabel) in zip(axes, METRICS):
        # Dataset stats per age group
        ds_m, ds_s = [], []
        for gname, (lo, hi) in AGE_BINS.items():
            vals = [r.get(key, np.nan) for r in dataset_results
                    if lo <= r["age"] < hi and not np.isnan(r.get(key, np.nan))]
            ds_m.append(np.mean(vals) if vals else np.nan)
            ds_s.append(np.std(vals) if vals else np.nan)

        # Generated stats
        gn_m = [gen_by_group[g].get(key, np.nan) for g in GENERATED_GROUPS]
        gn_s = [gen_by_group[g].get(f"{key}_std", np.nan) for g in GENERATED_GROUPS]

        w = 0.35
        ds_colors = ["#BBDEFB", "#90CAF9", "#42A5F5"]
        gn_colors = ["#C8E6C9", "#A5D6A7", "#66BB6A"]
        ax.bar(x - w / 2, ds_m, w, yerr=ds_s, capsize=4,
               color=ds_colors, alpha=0.9, label="Dataset")
        ax.bar(x + w / 2, gn_m, w, yerr=gn_s, capsize=4,
               color=gn_colors, alpha=0.9, label="Generated")
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        _ax_style(ax)

    fig.suptitle("Dataset vs Generated: Key Gait Metrics by Age Group",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")

# ── Plot 12: per-metric dot-plot (each subject, coloured by age group) ────────

def plot_dot_plots(dataset_results, out_path):
    METRICS = [
        ("speed",         "Walking Speed (m/s)"),
        ("cv_time",       "Stride Time CV (%)"),
        ("knee_rom",      "Knee ROM (°)"),
        ("hip_rom",       "Hip ROM (°)"),
    ]
    def group_color(age):
        for gname, (lo, hi) in AGE_BINS.items():
            if lo <= age < hi:
                return GROUP_COLORS[gname]
        return "#888888"

    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 5))
    for ax, (key, ylabel) in zip(axes, METRICS):
        for r in dataset_results:
            v = r.get(key, np.nan)
            if not np.isnan(v):
                ax.scatter(r["age"], v, color=group_color(r["age"]),
                           alpha=0.55, s=30, edgecolors="none")
        _regression_on_ax(ax, [r["age"] for r in dataset_results],
                          [r.get(key, np.nan) for r in dataset_results],
                          color="#333333")
        ax.set_xlabel("Age (years)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        _ax_style(ax)

    # legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=GROUP_COLORS[g],
                     label=f"{g.capitalize()} ({AGE_BINS[g][0]}–{AGE_BINS[g][1]}y)")
               for g in GENERATED_GROUPS]
    fig.legend(handles=handles, loc="upper right", ncol=3, fontsize=8)
    fig.suptitle("Gait Metrics by Subject (coloured by Age Group) — Dataset",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")

# ── Summary table ─────────────────────────────────────────────────────────────

def plot_violin_comparison(dataset_results, gen_gait_by_group, out_path):
    """
    Violin plot comparing dataset vs generated across age groups.

    Each age group gets two adjacent violins — dataset (solid, left) and
    generated (transparent outline, right) — sharing the same age-group
    colour so the pairing is immediately obvious.
    """
    METRICS = [
        ("speed",   "Walking Speed (m/s)"),
        ("cv_time", "Stride Time CV (%)"),
        ("hip_rom", "Hip ROM (°)"),
    ]
    # Colour per age group
    AGE_COLORS = {"young": "#2196F3", "mid": "#FF9800", "old": "#F44336"}
    W = 0.28       # violin half-width
    OFFSET = 0.22  # left/right shift from group centre

    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 6))

    for ax, (key, ylabel) in zip(axes, METRICS):
        xtick_pos, xtick_labels = [], []

        for gi, (gname, (lo, hi)) in enumerate(AGE_BINS.items()):
            color = AGE_COLORS[gname]

            # ── Dataset: one aggregated value per subject
            ds_vals = [r.get(key, np.nan) for r in dataset_results
                       if lo <= r["age"] < hi and not np.isnan(r.get(key, np.nan))]

            # ── Generated: one value per clip
            gen_vals = [m.get(key, np.nan) for m in gen_gait_by_group.get(gname, [])
                        if not np.isnan(m.get(key, np.nan))]

            x_ds  = gi - OFFSET
            x_gen = gi + OFFSET
            xtick_pos.append(gi)
            xtick_labels.append(gname.capitalize())

            for data, xpos, alpha, label in [
                (ds_vals,  x_ds,  0.80, "Dataset"),
                (gen_vals, x_gen, 0.30, "Generated"),
            ]:
                if len(data) < 3:
                    continue
                parts = ax.violinplot(
                    data,
                    positions=[xpos],
                    widths=W * 2,
                    showmedians=True,
                    showextrema=True,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(alpha)
                    pc.set_edgecolor(color if alpha > 0.5 else "black")
                    pc.set_linewidth(1.2)
                for part_name in ("cmins", "cmaxes", "cbars"):
                    if part_name in parts:
                        parts[part_name].set_edgecolor(
                            color if alpha > 0.5 else "black"
                        )
                        parts[part_name].set_linewidth(1.0)
                # Median line: always solid black, on top
                if "cmedians" in parts:
                    parts["cmedians"].set_edgecolor("black")
                    parts["cmedians"].set_linewidth(2.5)
                    parts["cmedians"].set_zorder(5)

        ax.set_xticks(xtick_pos)
        ax.set_xticklabels(xtick_labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
        _ax_style(ax)

    # Shared legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#888888", alpha=0.80, label="Dataset"),
        Patch(facecolor="#888888", alpha=0.30, edgecolor="black", label="Generated"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.9)
    fig.suptitle("Dataset vs Generated — Gait Metric Distributions by Age Group",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def save_summary_csv(dataset_results, gen_by_group, out_path):
    import csv
    KEYS = ["speed", "stride_length", "stride_time", "cadence",
            "cv_time", "cv_length", "knee_rom", "hip_rom"]

    rows = []
    for gname, (lo, hi) in AGE_BINS.items():
        gr = [r for r in dataset_results if lo <= r["age"] < hi]
        row = {"source": "Dataset", "group": gname, "n": len(gr)}
        for k in KEYS:
            vals = [r.get(k, np.nan) for r in gr if not np.isnan(r.get(k, np.nan))]
            row[f"{k}_mean"] = f"{np.mean(vals):.3f}" if vals else "N/A"
            row[f"{k}_sd"]   = f"{np.std(vals):.3f}"  if len(vals) > 1 else "N/A"
        rows.append(row)

    for g in GENERATED_GROUPS:
        res = gen_by_group.get(g, {})
        row = {"source": "Generated", "group": g, "n": res.get("n_clips", 0)}
        for k in KEYS:
            row[f"{k}_mean"] = f"{res.get(k, np.nan):.3f}" if not np.isnan(res.get(k, np.nan)) else "N/A"
            row[f"{k}_sd"]   = f"{res.get(f'{k}_std', np.nan):.3f}" if not np.isnan(res.get(f"{k}_std", np.nan)) else "N/A"
        rows.append(row)

    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  Saved: {out_path.name}")

# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("Loading metadata...")
    metadata = load_metadata()
    print(f"  {len(metadata)} subjects in metadata")

    # ── 1. Dataset
    print("\nAnalysing dataset (humanml3d_new_joints_6)...")
    dataset = analyze_dataset(metadata)

    print("\nDataset age distribution:")
    for gname, (lo, hi) in AGE_BINS.items():
        n = sum(1 for r in dataset if lo <= r["age"] < hi)
        print(f"  {gname}: {n} subjects")

    # ── 2. Dataset plots
    print("\nGenerating dataset plots...")
    plot_scatter_grid(dataset, OUT_DIR / "dataset" / "age_vs_metrics_scatter.png")
    plot_individual_heatmaps(dataset, OUT_DIR / "dataset")
    plot_age_group_heatmaps(dataset, OUT_DIR / "dataset")
    plot_correlation_matrix(dataset, OUT_DIR / "dataset" / "correlation_matrix.png")
    plot_violin_age_group(dataset, OUT_DIR / "dataset" / "violin_by_age_group.png")
    plot_speed_vs_stride(dataset, OUT_DIR / "dataset" / "speed_vs_stride_age.png")
    plot_sex_age_group_boxes(dataset, OUT_DIR / "dataset" / "box_sex_age_group.png")
    plot_dot_plots(dataset, OUT_DIR / "dataset" / "dot_plot_age_group.png")

    # ── 3. Generated clips
    print("\nAnalysing generated clips (motion1_walk_forward)...")
    walk_dir = GEN_DIR / "motion1_walk_forward"
    gen_by_group = {}
    gen_gait_by_group = {}  # per-clip lists for violin plot
    for ag in GENERATED_GROUPS:
        print(f"  {ag}...")
        rec, all_combined = analyze_generated(walk_dir, ag, max_batches=16)
        gen_by_group[ag] = rec
        gen_gait_by_group[ag] = all_combined
        spd = rec.get("speed", np.nan)
        cvt = rec.get("cv_time", np.nan)
        print(f"    n_clips={rec['n_clips']}, speed={spd:.3f} m/s, cv_time={cvt:.2f}%")

    # ── 4. Generated plots
    print("\nGenerating plots for generated clips...")
    plot_generated_bars(gen_by_group, OUT_DIR / "generated" / "metric_bars.png")
    plot_clinical_thresholds(gen_by_group, OUT_DIR / "generated" / "clinical_thresholds.png")
    plot_generated_heatmaps(gen_by_group, OUT_DIR / "generated" / "heatmap_by_age_group.png")

    # ── 5. Combined
    print("\nGenerating combined plots...")
    plot_dataset_vs_generated(dataset, gen_by_group, OUT_DIR / "combined" / "dataset_vs_generated.png")
    plot_violin_comparison(dataset, gen_gait_by_group, OUT_DIR / "combined" / "violin_comparison.png")
    save_summary_csv(dataset, gen_by_group, OUT_DIR / "combined" / "summary_table.csv")

    print(f"\nAll outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
