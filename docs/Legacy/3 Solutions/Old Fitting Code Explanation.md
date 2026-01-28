>[!info|right]
> **Part of a series on**
> ###### [[Dataset Overview|Van Criekinge]]
> ---
> **Solutions**
> * [[3 Solutions/New Marker Fitting|New Marker Fitting]]
> * [[3 Solutions/Old Fitting Code Explanation|Old Fitting Code]]
> * [[Tools/Visualization Setup|Visualization Setup]]

>[!info]
> ###### Old Fitting Code Explanation
> [Type::Code Documentation]
> [Script::fit_smpl_markers.py (legacy)]
> [Status::Superseded]
> [Lines::1868]

This document provides a **comprehensive line-by-line explanation** of the legacy `2_fit_smpl_markers.py` implementation, detailing its utilities, SMPL fitting logic, and architectural decisions.

```python fold:"2_fit_smpl_markers.py (old)"
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit SMPL to Van Criekinge C3D markers (no MoSh).
- Uses smplx SMPL layer
- Per-subject betas (shared), per-frame pose+trans
- Robust marker subset, small temporal smoothness
- Writes *_smpl_params.npz + fit report
"""
import os, json, math, argparse, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# ------------------------- Marker subset & joint map -------------------------
# SMPL 24-joint order used here:
SMPL_JOINTS = [
    "pelvis","L_hip","R_hip","spine1","L_knee","R_knee","spine2","L_ankle","R_ankle",
    "spine3","L_foot","R_foot","neck","L_collar","R_collar","head",
    "L_shoulder","R_shoulder","L_elbow","R_elbow","L_wrist","R_wrist","L_hand","R_hand"
]
J = {n:i for i,n in enumerate(SMPL_JOINTS)}

# Subset of anatomical markers found in your metadata (robust across subjects)
MARKER_TO_JOINT = {
    # Head / upper trunk
    "LFHD":J["head"], "RFHD":J["head"], "LBHD":J["head"], "RBHD":J["head"],
    "C7":J["neck"], "T10":J["spine2"], "CLAV":J["spine3"], "STRN":J["spine3"],

    # Pelvis triad
    "LASI":J["pelvis"], "RASI":J["pelvis"], "SACR":J["pelvis"],

    # Shoulders / elbows / wrists
    "LSHO":J["L_shoulder"], "RSHO":J["R_shoulder"],
    "LELB":J["L_elbow"], "RELB":J["R_elbow"],
    "LWRA":J["L_wrist"], "LWRB":J["L_wrist"],
    "RWRA":J["R_wrist"], "RWRB":J["R_wrist"],

    # Lower limbs
    "LKNE":J["L_knee"], "RKNE":J["R_knee"],
    "LANK":J["L_ankle"], "RANK":J["R_ankle"],

    "LHEE":J["L_ankle"],  "RHEE":J["R_ankle"],
    "LTOE":J["L_foot"],   "RTOE":J["R_foot"],
}

# ------------------------------ Utilities -----------------------------------
def rotmat_to_axis_angle(R):
    # R: (3,3)
    eps = 1e-8
    cos = (np.trace(R) - 1.0) * 0.5
    cos = np.clip(cos, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-6:
        return np.zeros(3, dtype=np.float32)
    rx = R[2,1] - R[1,2]
    ry = R[0,2] - R[2,0]
    rz = R[1,0] - R[0,1]
    axis = np.array([rx, ry, rz], dtype=np.float64)
    axis /= (2.0*np.sin(angle) + eps)
    return (axis * angle).astype(np.float32)

def estimate_root_init(markers_np, used_names):
    """
    markers_np: (T,K,3) subset already (no NaNs after prep_markers)
    used_names: list[str] names aligned with K
    Returns: axis-angle (3,), transl_init (T,3)
    """
    name2i = {n.upper(): i for i, n in enumerate(used_names)}
    need = ["LASI","RASI","SACR","C7","CLAV","STRN"]
    if not all(n in name2i for n in need):
        # Fallback
        aa = np.zeros(3, dtype=np.float32)
        t0 = np.zeros((markers_np.shape[0],3), dtype=np.float32)
        return aa, t0

    LASI = markers_np[:, name2i["LASI"], :]
    RASI = markers_np[:, name2i["RASI"], :]
    SACR = markers_np[:, name2i["SACR"], :]
    C7   = markers_np[:, name2i["C7"],   :]
    CLAV = markers_np[:, name2i["CLAV"], :]
    STRN = markers_np[:, name2i["STRN"], :]

    pelvis = (LASI + RASI + SACR) / 3.0    # (T,3)
    upper  = (C7 + CLAV + STRN) / 3.0      # (T,3)

    # Pick first frame
    p0 = pelvis[0]; u0 = upper[0]
    up = u0 - p0; up /= (np.linalg.norm(up) + 1e-8)
    lr = LASI[0] - RASI[0]; lr /= (np.linalg.norm(lr) + 1e-8)  # left minus right
    fwd = np.cross(up, lr); fwd /= (np.linalg.norm(fwd) + 1e-8)
    lr  = np.cross(fwd, up); lr  /= (np.linalg.norm(lr) + 1e-8)

    # Columns: X=lr, Y=up, Z=fwd → world
    Rw = np.stack([lr, up, fwd], axis=1)          # 3x3
    aa = rotmat_to_axis_angle(Rw)                 # (3,)
    return aa.astype(np.float32), pelvis.astype(np.float32)

def huber(x, delta=0.01):
    a = x.abs()
    mask = (a <= delta).float()
    return 0.5*(a**2)*mask + delta*(a-0.5*delta)* (1.0-mask)

def linear_interp_nans(arr):  # (T,3)
    x = arr.copy()
    T = x.shape[0]
    for d in range(3):
        v = x[:,d]
        nans = np.isnan(v)
        if nans.any():
            idx = np.arange(T)
            v[nans] = np.interp(idx[nans], idx[~nans], v[~nans])
        x[:,d] = v
    return x

def decimate_indices(T, src_fps, dst_fps):
    if src_fps == dst_fps: return np.arange(T)
    step = src_fps / dst_fps
    return np.clip(np.round(np.arange(0, T, step)).astype(int), 0, T-1)

def sanitize(name: str):
    return name.replace(" ", "_").replace("(", "").replace(")", "")

# ------------------------------ Fitter --------------------------------------
class SMPLFitter(nn.Module):
    def __init__(self, smpl_layer, T, device="cpu", init_betas=None):
        super().__init__()
        self.smpl = smpl_layer
        self.device = device
        # Parameters
        self.global_orient = nn.Parameter(torch.zeros(T, 3, device=device))
        self.transl = nn.Parameter(torch.zeros(T, 3, device=device))
        self.body_pose = nn.Parameter(torch.zeros(T, 69, device=device))   # axis-angle 23*3
        betas_init = torch.zeros(10, device=device) if init_betas is None else torch.as_tensor(init_betas, device=device).float()
        self.betas = nn.Parameter(betas_init)

    def forward(self):
        # Returns joints (T,24,3)
        out = self.smpl(
            global_orient=self.global_orient,
            body_pose=self.body_pose,
            betas=self.betas.unsqueeze(0).expand(self.global_orient.shape[0], -1),
            transl=self.transl,
            pose2rot=True
        )
        return out.joints  # (T, 24, 3)

def build_smpl(model_dir: str, gender: str, device: str):
    import smplx
    # smplx.create expects directory containing SMPL_*.pkl
    smpl = smplx.create(model_path=model_dir, model_type="smpl", gender=gender,
                        use_pca=False, num_betas=10).to(device)
    smpl.eval()
    for p in smpl.parameters(): p.requires_grad = False
    return smpl

def pick_gender(subject_meta):
    g = subject_meta.get("gender","neutral").lower()
    if g in ("male","female"): return g
    return "neutral"

def load_markers_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    M = d["marker_data"]        # (T, n_markers, 3) in meters
    names = [str(x) for x in d["marker_names"].tolist()]
    fps = float(d["frame_rate"])
    return M, names, fps

def subset_markers(M, names):
    keep = []
    targets = []
    used = []
    for i, nm in enumerate(names):
        key = nm.strip().upper()
        if key in MARKER_TO_JOINT:
            keep.append(i)
            targets.append(MARKER_TO_JOINT[key])
            used.append(key)
    if len(keep) < 12:
        raise RuntimeError("Too few anatomical markers found after subsetting.")
    return M[:, keep, :], used, np.array(targets, dtype=np.int64)

def prep_markers(M_sub, max_gap=4, drop_thresh=0.6):
    # Interp small gaps, drop frames with > (1-drop_thresh) missing
    T, K, _ = M_sub.shape
    valid = ~np.isnan(M_sub).any(axis=2)
    keep_frames = (valid.sum(axis=1) >= math.ceil(drop_thresh*K))
    M2 = M_sub[keep_frames].copy()
    # interpolate independently per marker
    for k in range(M2.shape[1]):
        M2[:,k,:] = linear_interp_nans(M2[:,k,:])
    return M2, keep_frames

def subject_cache_paths(out_root, subject_id):
    subj_dir = Path(out_root) / subject_id
    subj_dir.mkdir(parents=True, exist_ok=True)
    return subj_dir, subj_dir / "betas.npy"

def fit_sequence(smpl, markers_torch, marker_joint_idx, used_names, init_betas, device, iters=400,
                 w_marker=1.0, w_pose=1e-3, w_betas=5e-4, w_vel=1e-2, w_acc=1e-3):
    T, K, _ = markers_torch.shape
    fitter = SMPLFitter(smpl, T, device, init_betas=init_betas).to(device)

    mk_np = markers_torch.detach().cpu().numpy()     # (T,K,3)
    aa, pelvis_series = estimate_root_init(mk_np, used_names)  # (3,), (T,3)
    fitter.global_orient.data[:] = torch.from_numpy(np.repeat(aa[None, :], T, axis=0)).to(device)
    fitter.transl.data[:]        = torch.from_numpy(pelvis_series).to(device)

    opt = Adam([
        {"params":[fitter.global_orient, fitter.body_pose, fitter.transl], "lr":3e-2},
        {"params":[fitter.betas], "lr":1e-3}
    ])
    hub = nn.SmoothL1Loss(beta=0.01, reduction='none')  # Huber-like

    targets_idx = torch.as_tensor(marker_joint_idx, device=device, dtype=torch.long)  # (K,)
    for it in range(iters):
        opt.zero_grad()
        joints = fitter()                         # (T,24,3)
        # gather joint positions for each marker
        pick = joints[:, targets_idx, :]          # (T,K,3)
        names = np.array(used_names)                          # list[str] length K
        w = np.ones(len(names), dtype=np.float32)
        leg_keys = {"LKNE","RKNE","LANK","RANK","LHEE","RHEE","LTOE","RTOE"}
        w[[i for i, n in enumerate(names) if n in leg_keys]] = 1.5
        w_t = torch.from_numpy(w).to(device)[None, :, None]   # (1,K,1)

        per_elem = hub(pick, markers_torch)                   # (T,K,3), reduction='none'
        marker_loss = (per_elem * w_t).mean()

        # priors & smoothness
        pose_l2 = (fitter.body_pose**2).mean()
        betas_l2 = (fitter.betas**2).mean()
        vel = (fitter.transl[1:] - fitter.transl[:-1]).pow(2).mean()
        acc = (fitter.transl[2:] - 2*fitter.transl[1:-1] + fitter.transl[:-2]).pow(2).mean() if T>2 else torch.tensor(0., device=device)

        loss = w_marker*marker_loss + w_pose*pose_l2 + w_betas*betas_l2 + w_vel*vel + w_acc*acc
        loss.backward()
        opt.step()

        if (it+1) % 100 == 0 or it==0:
            print(f"  iter {it+1:03d}: total={loss.item():.6f}  marker={marker_loss.item():.6f}")

    with torch.no_grad():
        joints = fitter()
    result = {
        "poses": fitter.body_pose.detach().cpu().numpy(),         # (T,63)
        "global_orient": fitter.global_orient.detach().cpu().numpy(),  # (T,3)
        "trans": fitter.transl.detach().cpu().numpy(),            # (T,3)
        "betas": fitter.betas.detach().cpu().numpy(),             # (10,)
        "joints": joints.detach().cpu().numpy(),                  # (T,24,3)
        "final_marker_loss": float(marker_loss.detach().cpu().item())
    }
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", required=True, help=".../data/processed/van_criekinge")
    ap.add_argument("--models_dir", required=True, help=".../body_models/smpl (contains SMPL_*.pkl)")
    ap.add_argument("--out_dir", required=True, help="where to write smpl_fitted_smpl")
    ap.add_argument("--subject", default=None, help="e.g., SUBJ01 (default: all subjects under processed_dir)")
    ap.add_argument("--trial", default=None, help="e.g., 'SUBJ1 (0)' (optional: fit only this trial)")
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--iters", type=int, default=400)
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    subjects = [args.subject] if args.subject else sorted([p.name for p in processed.iterdir() if p.is_dir()])
    print(f"Found {len(subjects)} subjects")

    for subj in subjects:
        subj_dir = processed / subj
        npz_files = sorted(glob.glob(str(subj_dir / "*_markers_positions.npz")))
        if not npz_files:
            print(f"[{subj}] no *_markers_positions.npz files — skipping")
            continue
        # cache paths
        subj_out_dir, betas_path = subject_cache_paths(out_root, subj)
        # gender read from any metadata file
        meta_files = sorted(glob.glob(str(subj_dir / "*_metadata.json")))
        gender = "neutral"
        subj_meta = {}
        if meta_files:
            with open(meta_files[0], "r") as f:
                subj_meta = json.load(f)
            gender = pick_gender(subj_meta)
        print(f"[{subj}] gender={gender}")

        # build SMPL for gender (or neutral fallback)
        device = "cuda" if (args.device=="cuda" and torch.cuda.is_available()) else "cpu"
        smpl = build_smpl(args.models_dir, gender, device)

        # Load/initialize betas
        init_betas = None
        if betas_path.exists():
            init_betas = np.load(betas_path)
            print(f"[{subj}] loaded cached betas {betas_path}")
        else:
            print(f"[{subj}] no cached betas; will optimize and save")

        # Optionally filter to one trial
        if args.trial:
            npz_files = [p for p in npz_files if Path(p).stem.startswith(args.trial)]
            if not npz_files:
                print(f"[{subj}] requested trial {args.trial} not found")
                continue

        # If no betas, fit them once on the first trial (or only trial)
        if init_betas is None:
            first_npz = npz_files[0]
            M, names, fps = load_markers_npz(first_npz)
            M_sub, used_names, tgt_idx = subset_markers(M, names)
            M_sub, keep = prep_markers(M_sub, max_gap=4, drop_thresh=0.6)
            T = M_sub.shape[0]
            if T > 220:   # limit for fast betas fit
                sel = decimate_indices(T, src_fps=fps, dst_fps=min(fps, 20))
                M_sub = M_sub[sel]
                T = M_sub.shape[0]
            markers_torch = torch.from_numpy(M_sub).float().to(device)
            result = fit_sequence(smpl, markers_torch, tgt_idx, used_names, init_betas=None, device=device, iters=min(args.iters*2, 800))
            np.save(betas_path, result["betas"])
            init_betas = result["betas"]
            print(f"[{subj}] saved betas → {betas_path}")

        # Fit each trial with cached betas
        for npz_path in npz_files:
            trial_name = Path(npz_path).stem.replace("_markers_positions","")
            trial_safe = sanitize(trial_name)
            out_trial = subj_out_dir / f"{trial_safe}_smpl_params.npz"
            report_path = subj_out_dir / f"{trial_safe}_smpl_metadata.json"

            if out_trial.exists():
                print(f"[{subj}] {trial_name}: already fitted → {out_trial.name}")
                continue

            M, names, fps = load_markers_npz(npz_path)
            M_sub, used_names, tgt_idx = subset_markers(M, names)
            M_sub, keep = prep_markers(M_sub, max_gap=4, drop_thresh=0.6)
            markers_torch = torch.from_numpy(M_sub).float().to(device)

            print(f"[{subj}] fitting {trial_name}: frames={M_sub.shape[0]} fps={fps} markers={len(used_names)}")
            result = fit_sequence(smpl, markers_torch, tgt_idx, used_names, init_betas=init_betas, device=device, iters=args.iters)

            # Pack SMPL params (HumanML3D expects body_pose+global_orient concatenated to 72D)
            poses72 = np.concatenate([result["global_orient"], result["poses"]], axis=1)  # (T,72)
            np.savez(out_trial,
                     poses=poses72, trans=result["trans"], betas=result["betas"],
                     gender=gender, subject_id=subj, trial_name=trial_name, fps=fps,
                     n_frames=result["trans"].shape[0], joints=result["joints"])

            report = {
                "subject_id": subj, "trial_name": trial_name, "gender": gender,
                "frames_kept": int(markers_torch.shape[0]), "fps": fps,
                "markers_used": used_names,
                "final_marker_loss": result["final_marker_loss"]
            }
            with open(report_path, "w") as f: json.dump(report, f, indent=2)
            print(f"[{subj}] wrote {out_trial.name} and {report_path.name}")

if __name__ == "__main__":
    main()
```

Overall, this takes cleaned marker trajectories from step 1 and fit a SMPL body such that
- Per subject, one shared body shape vector `betas`
- Per frame
	- pose (`global_orient` + `body_pose`) 
	- root translation (`transl`)

Then it will write pre-trial outputs
`poses` (Tx72) = `[global_orient` + `body_pose`) and root translation (`transl`)
`trans` (Tx3)
`betas` (10,)
`joints` (Tx24x3) from SMPL **(24 joints)**
with metadata and JSON report

>[!FACT] It's just a simple marker to joint fitting with some translation smoothing

---

# Key Definitions

## SMPL Joint Order

In the code, there is this

```python
SMPL_JOINTS = ["pelvis", "L_hip", "R_hip", ..., "R_hand"] # Length 24, gives each joint a name in a specific order
J = {name: index} # Creates lookup tables to index.
```

This defines the 24 joint indexing used everywhere.
- `{python}J["pelvis"] == 0`
- `{python}J["L_knee"] == 4`

## Marker Subset Mapping

`{python}MARKER_TO_JOINT` maps marker label to SMPL joint index

Example:
- `"LASI" -> pelvis joint`
- `"LKNE" -> L_knee joint`
- `"LTOE" -> L_foot joint`

Importantly, multiple markers map to the same SMPL joint
- All 4 head markers maps to `head`
- CLAV + STRN maps to `spine3`
- LHEE + LANK maps to `L_ankle`

In the script, there are N total marker labels in NPZ, but we don't use all. We loop through N markers and keep a K size one of them. Multiple of these K markers map to the same SMPL joint.

---

# Utilities

## `rotmat_to_axis_angle(R)`

> [!summary] Purpose
> Convert a 3×3 rotation matrix into an axis-angle (Rodrigues) 3-vector (the pose format `smplx` expects).

> [!info] I/O
> **Input:**
>
> * `R`: (3,3) rotation matrix (NumPy)
>
> **Output:**
>
> * `(3,)` axis-angle vector (float32)

> [!example]- Steps
>
> * Compute rotation angle from the trace:
>
>   * `cos = (trace(R) - 1) / 2`
>   * `angle = arccos(clamp(cos, -1, 1))`
> * If `angle` is tiny → return `[0,0,0]`
> * Otherwise compute “skew” terms:
>
>   * `rx = R[2,1] - R[1,2]`
>   * `ry = R[0,2] - R[2,0]`
>   * `rz = R[1,0] - R[0,1]`
> * Form axis vector `axis = [rx, ry, rz]`
> * Normalize:
>
>   * `axis /= (2*sin(angle) + eps)`
> * Return `axis * angle` as float32

Used only in `estimate_root_init`.

---

## `estimate_root_init(markers_np, used_names)`

> [!summary] Purpose
> Build a crude initialization for the SMPL root:
>
> * `global_orient`: one axis-angle, copied to all frames
> * `transl`: per-frame translation initialized from pelvis markers

> [!info] I/O
> **Inputs:**
>
> * `markers_np`: (T, K, 3) marker positions after subsetting + prep (intended: no NaNs)
> * `used_names`: list length K, marker names aligned with axis 1
>
> **Outputs:**
>
> * `aa`: (3,) axis-angle initial orientation
> * `pelvis_series`: (T,3) initial translations

> [!example]- Steps
>
> 1. Build marker name → index:
>
>    ```python
>    name2i = {n.upper(): i for i,n in enumerate(used_names)}
>    ```
> 2. Require pelvis + trunk markers:
>
>    * `LASI, RASI, SACR, C7, CLAV, STRN`
>    * If any missing: return `aa = 0`, `pelvis_series = zeros(T,3)` (weak fallback)
> 3. Extract trajectories (each is (T,3)):
>
>    * `LASI(t), RASI(t), SACR(t), C7(t), CLAV(t), STRN(t)`
> 4. Compute pelvis + upper-trunk centroids:
>
>    * `pelvis(t) = (LASI + RASI + SACR)/3`
>    * `upper(t)  = (C7 + CLAV + STRN)/3`
> 5. Use **frame 0 only** to define body axes:
>
>    * `up  = normalize(upper[0] - pelvis[0])`
>    * `lr  = normalize(LASI[0] - RASI[0])`  (left-minus-right)
>    * `fwd = normalize(cross(up, lr))`
>    * re-orthogonalize: `lr = normalize(cross(fwd, up))`
> 6. Build a rotation matrix with columns:
>
>    * X = `lr`, Y = `up`, Z = `fwd`
>    * `Rw = [lr up fwd]` (3×3)
> 7. Convert `Rw → aa` via `rotmat_to_axis_angle`
> 8. Return:
>
>    * `aa`
>    * `pelvis_series = pelvis(t)`

This implies:

* Orientation comes from **one frame** (frame 0).
* Assumes marker-derived axes are a valid “world/body alignment”.
* **No coordinate conversion** (Vicon vs SMPL conventions), which is why pelvis can end up sideways.

---

## `huber(x, delta=0.01)`

> [!summary] Purpose
> Compute a Huber-style robust penalty for a tensor, elementwise. (Applied to every individual number in the tensor before average/sum)
> - Quadratic loss for small errors (behaves like MSE and encourages precision)
> - Linear for big errors (Huge outliers don't dominate the fit)
> 
> For error $e$:
> $$
> \text{Huber}(e) = \begin{cases}
> \frac{1}{2} e^{2} & \text{if } |e| < \delta \\
> \delta\left( |e| - \frac{1}{2} \delta \right) & \text{if } |e| > \delta 
> \end{cases}
> $$

> [!info] I/O
> **Input:**
>
> * `x`: torch tensor of any shape
> * `delta`: threshold between L2 and L1 regions
>
> **Output:**
>
> * torch tensor same shape as `x`, containing per-element Huber penalty

> [!example]- Steps
>
> * Let `a = abs(x)`
> * If `a <= delta`: use quadratic region `0.5 * a^2`
> * Else: use linear region `delta*(a - 0.5*delta)`
> * Return piecewise result

⚠️ In this script, this function is **never used** (the code uses `nn.SmoothL1Loss` instead).

---

## `linear_interp_nans(arr)`

> [!summary] Purpose
> Gap filler: Fill missing (NaN) values in a (T,3) trajectory using linear interpolation over time.
> Take one coordinate over time, find where it's NaN and fill by drawing a straight line.

> [!info] I/O
> **Input:**
>
> * `arr`: (T,3) NumPy array, may contain NaNs
>
> **Output:**
>
> * (T,3) NumPy array with NaNs replaced by interpolated values

> [!example]- Steps
>
> * Copy input to `x`
> * For each dimension `d ∈ {0,1,2}`:
>
>   * `v = x[:, d]`
>   * find `nans = isnan(v)`
>   * if any NaNs:
>
>     * build indices `idx = arange(T)`
>     * fill `v[nans] = interp(idx[nans], idx[~nans], v[~nans])`
>   * write back to `x[:, d]`
> * Return `x`

Used in `prep_markers` (per marker, per coordinate).

---

## `decimate_indices(T, src_fps, dst_fps)`

> [!summary] Purpose
> Generate frame indices to **downsample** a sequence from `src_fps` to `dst_fps` by nearest-frame selection.

> [!info] I/O
> **Inputs:**
>
> * `T`: number of frames in the original sequence
> * `src_fps`: original sampling rate
> * `dst_fps`: target sampling rate
>
> **Output:**
>
> * NumPy int array of frame indices into `[0, T-1]`

> [!example]- Steps
>
> * If `src_fps == dst_fps`: return `[0,1,2,...,T-1]`
> * Else:
>
>   * `step = src_fps / dst_fps`
>   * sample times: `0, step, 2*step, ...`
>   * round to nearest integer frame index
>   * clip into `[0, T-1]`

Used only during the “fit betas fast” phase (limits frames for speed).

---

## `sanitize(name)`

> [!summary] Purpose
> Convert a trial name into a filename-friendly string.

> [!info] I/O
> **Input:**
>
> * `name`: string (e.g., `"SUBJ1 (1)"`)
>
> **Output:**
>
> * sanitized string (e.g., `"SUBJ1_1"`)

> [!example]- Steps
>
> * Replace spaces with underscores
> * Remove `(` and `)`

Used when naming output files like `SUBJ1_1_smpl_params.npz`.

---

# The SMPL model and fitting core

## `class SMPLFitter(nn.Module)`

> [!summary] Purpose
> Hold the **trainable SMPL parameters** (pose, translation, shape) for an entire sequence, and provide a `forward()` that returns the corresponding **SMPL joint positions** for each frame.
>
> a "differentiable container” around `smplx.create(..., model_type="smpl")`:
>
> * SMPL model weights are frozen.
> * This module stores the *inputs* you optimize to match markers.

> [!info] I/O
> **Inputs (conceptual):**
>
> * A frozen `smplx` SMPL layer (`smpl_layer`)
> * Sequence length `T`
> * Optional subject shape initialization `init_betas`
>
> **Outputs (from `forward`)**
>
> * `joints`: (T, 24, 3) SMPL joint positions, in world coordinates (meters), with translation applied

> [!example]- Internal state (trainable parameters)
> Stored as `torch.nn.Parameter` so Adam can optimize them:
>
> * `global_orient`: (T, 3)
>
>   * Root orientation per frame in axis-angle form
>   * This rotates the entire body each frame
> * `body_pose`: (T, 69)
>
>   * 23 body joints × 3 axis-angle parameters each
>   * “Pose” excluding the root joint
> * `transl`: (T, 3)
>
>   * Global translation of the root each frame
>   * In this script it’s initialized to the pelvis marker centroid and then smoothed
> * `betas`: (10,)
>
>   * Subject shape parameters
>   * Shared across all frames (constant body shape for the subject)

---

### `__init__(smpl_layer, T, device="cpu", init_betas=None)`

> [!summary] Purpose
> Construct a per-sequence parameter set for SMPL fitting:
>
> * one set of pose+translation per frame
> * one shared shape (`betas`) for the subject

> [!info] I/O
> **Inputs:**
>
> * `smpl_layer`: an already-created SMPL model from `smplx` (frozen weights)
> * `T`: integer number of frames
> * `device`: `"cpu"` or `"cuda"`
> * `init_betas`: optional (10,) NumPy or torch array
>
> **Output:**
>
> * A module instance containing trainable parameters

> [!example]- Steps
>
> 1. Store handles:
>
>    * `self.smpl = smpl_layer`
>    * `self.device = device`
> 2. Allocate trainable per-frame parameters initialized to zero:
>
>    * `global_orient = zeros(T,3)`
>    * `transl       = zeros(T,3)`
>    * `body_pose    = zeros(T,69)`
> 3. Allocate shared shape:
>
>    * if `init_betas is None`: `betas = zeros(10)`
>    * else: `betas = tensor(init_betas).float()`
> 4. Wrap each as `nn.Parameter` so gradients are tracked and Adam can update them

> [!note] Practical implications
>
> * Starting from zeros means the initial body is roughly a SMPL T-pose (after optimization starts), but *this script* overwrites `global_orient` and `transl` immediately using `estimate_root_init`.
> * Shape (`betas`) is either:
>
>   * optimized from scratch (first time for a subject), or
>   * loaded from cache and kept as a starting point.

---

### `forward()`

> [!summary] Purpose
> Run the SMPL layer using the current parameters and return **3D joint positions** for every frame.

> [!info] I/O
> **Input:**
>
> * none explicitly (uses internal parameters)
>
> **Output:**
>
> * `out.joints`: (T, 24, 3)

> [!example]- Steps
>
> 1. Expand `betas` from (10,) to (T,10) because `smplx` expects per-frame betas:
>
>    ```python
>    betas_T = self.betas.unsqueeze(0).expand(T, -1)  # (T,10)
>    ```
> 2. Call the frozen SMPL model:
>
>    ```python
>    out = self.smpl(
>      global_orient=self.global_orient,  # (T,3)
>      body_pose=self.body_pose,          # (T,69)
>      betas=betas_T,                     # (T,10)
>      transl=self.transl,                # (T,3)
>      pose2rot=True
>    )
>    ```
> 3. Return joints:
>
>    ```python
>    return out.joints  # (T,24,3)
>    ```

> [!note] Important details
>
> * `pose2rot=True` means the SMPL layer will convert axis-angle → rotation matrices internally.
> * `out.joints` are in the same coordinate system as `transl` + the SMPL model output.
>   If your marker coordinate system is different (Vicon Z-up, etc.), then fitting quality will suffer unless you convert coordinates before calling this.

---

## `build_smpl(model_dir: str, gender: str, device: str)`

> [!summary] Purpose
> Create a **frozen SMPL body model** (via `smplx`) that the fitter will call during optimization.
>
> This is the “renderer” that maps `(global_orient, body_pose, betas, transl)` → `joints`.

> [!info] I/O
> **Inputs:**
>
> * `model_dir`: path to SMPL model files (e.g., containing `SMPL_NEUTRAL.pkl`, etc.)
> * `gender`: `"male"`, `"female"`, or `"neutral"`
> * `device`: `"cpu"` or `"cuda"`
>
> **Output:**
>
> * `smpl`: a `smplx` SMPL model instance on the selected device

> [!example]- Steps
>
> 1. Import `smplx` locally (so the script can import without smplx installed until needed):
>
>    ```python
>    import smplx
>    ```
> 2. Create the model:
>
>    ```python
>    smpl = smplx.create(
>      model_path=model_dir,
>      model_type="smpl",
>      gender=gender,
>      use_pca=False,
>      num_betas=10
>    ).to(device)
>    ```
> 3. Put the model in eval mode:
>
>    * `smpl.eval()`
> 4. Freeze all SMPL internal parameters:
>
>    ```python
>    for p in smpl.parameters():
>        p.requires_grad = False
>    ```
> 5. Return the frozen model

> [!note] Why freeze it?
>
> * You want optimization to adjust **pose/shape/trans**, not the model’s learned/template parameters.
> * Freezing reduces memory usage and prevents “cheating” by warping the model.

> [!note] Why `use_pca=False`?
>
> * In SMPL-X ecosystems, PCA is sometimes used for hand pose compression.
> * Here you’re using “full” axis-angle style pose parameters rather than a reduced PCA space, which is simpler for optimization.

---

## `pick_gender(subject_meta)`

> [!summary] Purpose
> Decide which SMPL gender model to use for a subject based on their metadata, with a safe fallback.

> [!info] I/O
> **Input:**
>
> * `subject_meta`: dict loaded from your subject/trial metadata JSON
>
> **Output:**
>
> * `"male"`, `"female"`, or `"neutral"`

> [!example]- Steps
>
> 1. Read gender string from metadata:
>
>    ```python
>    g = subject_meta.get("gender", "neutral").lower()
>    ```
> 2. If it matches `"male"` or `"female"`, return it.
> 3. Otherwise return `"neutral"`.

> [!note] Practical implications
>
> * If your metadata is inconsistent or missing, you won’t crash; you’ll fit the neutral model.
> * Gendered models can slightly change body proportions and joint placements; using the “wrong” gender generally won’t destroy fitting, but can slightly increase residual error.

---

## How these pieces connect in the fitting loop (quick mental model)

> [!example]- Data flow (core model path)
>
> * `build_smpl(...)` creates a frozen SMPL layer.
> * `SMPLFitter(...)` creates trainable tensors `global_orient, body_pose, transl, betas`.
> * Each optimization iteration calls:
>
>   * `joints = fitter()` → (T,24,3)
> * The script then selects a subset of joints to compare against marker positions:
>
>   * `pick = joints[:, targets_idx, :]` → (T,K,3)
> * Marker loss + priors → gradients → Adam updates those parameters.
> * SMPL weights never change; only `SMPLFitter` parameters move.

# Marker I/O and preprocessing

## `load_markers_npz(npz_path)`

> [!summary] Purpose
> Load one preprocessed marker file (`*_markers_positions.npz`) produced by Step 1, and return the marker tensor + labels + sampling rate in a consistent format.

> [!info] I/O
> **Input:**
>
> * `npz_path`: path to `..._markers_positions.npz`
>
> **Output:**
>
> * `M`: (T, N, 3) NumPy float array — marker positions in **meters**
> * `names`: list[str] length N — marker labels aligned with axis 1 of `M`
> * `fps`: float — marker frame rate

> [!example]- Steps
>
> 1. Load NPZ with pickle allowed:
>
>    ```python
>    d = np.load(npz_path, allow_pickle=True)
>    ```
> 2. Extract marker tensor:
>
>    * `M = d["marker_data"]`  → (T, N, 3)
> 3. Extract labels:
>
>    * `names = [str(x) for x in d["marker_names"].tolist()]`
> 4. Extract sampling rate:
>
>    * `fps = float(d["frame_rate"])`
> 5. Return `(M, names, fps)`

> [!note] Alignment invariant
> `names[i]` corresponds to `M[:, i, :]` for all frames.

---

## `subset_markers(M, names)`

> [!summary] Purpose
> Select a robust subset of “anatomical” markers and build a per-marker target SMPL joint index, so the fitter can compare:
>
> * predicted joint positions (from SMPL)
> * against marker positions (from C3D)

> [!info] I/O
> **Inputs:**
>
> * `M`: (T, N, 3) marker positions
> * `names`: list[str] length N
>
> **Outputs:**
>
> * `M_sub`: (T, K, 3) marker subset (K ≤ N)
> * `used_names`: list[str] length K, marker labels (uppercased)
> * `marker_joint_idx`: (K,) NumPy int64 array, each entry ∈ [0..23] (SMPL joint id)

> [!example]- Steps
>
> 1. Initialize empty lists:
>
>    * `keep = []` (indices into the original marker axis)
>    * `targets = []` (SMPL joint index for each kept marker)
>    * `used = []` (cleaned marker names)
> 2. For each marker index `i` and name `nm`:
>
>    * `key = nm.strip().upper()`
>    * If `key in MARKER_TO_JOINT`:
>
>      * append `i` to `keep`
>      * append `MARKER_TO_JOINT[key]` to `targets`
>      * append `key` to `used`
> 3. Validate marker count:
>
>    * If `< 12` markers found → raise error
>      (script assumes too few markers makes fitting unstable)
> 4. Slice the marker tensor:
>
>    * `M_sub = M[:, keep, :]`  → (T, K, 3)
> 5. Return:
>
>    * `M_sub`
>    * `used`
>    * `np.array(targets, dtype=np.int64)`

> [!note] Key idea
> This function creates the “matching plan” for fitting:
>
> * marker `k` in `M_sub[:, k, :]` should be compared to SMPL joint `marker_joint_idx[k]`.

> [!warning] Many-to-one mapping
> Multiple markers can map to the same SMPL joint (e.g., LFHD/RFHD/LBHD/RBHD → head).
> That means you’re fitting “clusters” of markers to one joint position, which can bias the fit if not weighted carefully.

---

## `prep_markers(M_sub, max_gap=4, drop_thresh=0.6)`

> [!summary] Purpose
> Clean the time series after subsetting by:
>
> 1. Dropping frames with “too many missing markers”
> 2. Filling remaining NaNs via per-marker linear interpolation over time

> [!info] I/O
> **Inputs:**
>
> * `M_sub`: (T, K, 3) marker positions, may contain NaNs
> * `max_gap`: (int) *intended* maximum gap length allowed for interpolation (**unused in this version**)
> * `drop_thresh`: fraction of markers that must be present in a frame to keep it
>
> **Outputs:**
>
> * `M2`: (T_kept, K, 3) cleaned marker tensor
> * `keep_frames`: (T,) boolean mask indicating which original frames survived

> [!example]- Steps
>
> 1. Compute validity mask per frame and marker:
>
>    ```python
>    valid = ~np.isnan(M_sub).any(axis=2)   # (T, K)
>    ```
>
>    * `valid[t,k] == True` means marker k has all xyz on frame t
> 2. Decide which frames to keep:
>
>    ```python
>    keep_frames = (valid.sum(axis=1) >= ceil(drop_thresh * K))  # (T,)
>    ```
>
>    * With `drop_thresh=0.6`, a frame must have ≥ 60% valid markers.
> 3. Drop the bad frames:
>
>    ```python
>    M2 = M_sub[keep_frames].copy()  # (T_kept, K, 3)
>    ```
> 4. For each marker `k` independently, fill NaNs over time:
>
>    ```python
>    for k in range(K):
>        M2[:, k, :] = linear_interp_nans(M2[:, k, :])  # (T_kept, 3)
>    ```
> 5. Return `(M2, keep_frames)`

> [!note] What gets “smoothed” here
> This is not smoothing—this is **gap filling**:
>
> * It only changes values where NaNs existed.
> * It does not apply filtering to valid data.

> [!warning] Two gotchas in this implementation
>
> * `max_gap` is unused: long gaps will still be interpolated.
> * If a marker is NaN for *all* kept frames, `np.interp` can fail (depends on how many valid samples remain).
>   The frame-dropping step reduces the chance, but doesn’t mathematically guarantee safety.

---

## `subject_cache_paths(out_root, subject_id)`

> [!summary] Purpose
> Create (if needed) and return the per-subject output directory and the path used to cache that subject’s fitted SMPL shape (`betas.npy`).

> [!info] I/O
> **Inputs:**
>
> * `out_root`: root output directory (string or Path)
> * `subject_id`: e.g. `"SUBJ01"`
>
> **Outputs:**
>
> * `subj_dir`: Path to subject output folder (`out_root/subject_id/`)
> * `betas_path`: Path to cached shape file (`out_root/subject_id/betas.npy`)

> [!example]- Steps
>
> 1. Build the subject folder path:
>
>    ```python
>    subj_dir = Path(out_root) / subject_id
>    ```
> 2. Ensure it exists:
>
>    ```python
>    subj_dir.mkdir(parents=True, exist_ok=True)
>    ```
> 3. Define cache path:
>
>    ```python
>    betas_path = subj_dir / "betas.npy"
>    ```
> 4. Return `(subj_dir, betas_path)`

> [!note] Why cache `betas`?
> Shape is assumed constant for a subject.
> So the script fits `betas` once (usually on the first trial), saves it, and reuses it for other trials to improve stability and reduce compute.

# The optimization:

## `fit_sequence(...)`

> [!summary] Purpose
> Fit SMPL parameters to marker trajectories by optimizing:
>
> * **Per-frame** root orientation `global_orient[t]`, pose `body_pose[t]`, translation `transl[t]`
> * **Per-subject** shape `betas` (shared across all frames)
>
> The optimizer tries to make SMPL joint positions match observed markers while keeping motion “reasonable”.

---

### Mental model

> [!abstract] What’s happening conceptually
> This function creates a differentiable system:
>
> **Parameters → SMPL → joints → loss(marker mismatch + priors) → gradients → Adam update**
>
> You can imagine it like:
>
> ```text
> (global_orient[t], body_pose[t], transl[t], betas)
>            │
>            ▼
>      SMPL forward
>            │
>            ▼
>     joints[t, j, :]
>            │
>            ├─ gather joints for each marker mapping
>            ▼
>     pick[t, k, :]
>            │
>            ▼
>  compare to markers[t, k, :]
>            │
>            ▼
>         total loss
>            │
>            ▼
>        backprop + Adam
> ```

> [!note] What “differentiable system” means here
> A **differentiable system** is a chain of computations where changing the inputs a tiny bit changes the outputs smoothly, so we can compute **gradients** (derivatives) that tell us *which direction to change the inputs to reduce the loss*.
>
> In `fit_sequence`, the “system” is:
>
> ```text
> (global_orient, body_pose, transl, betas)
>            ↓
>         SMPL (smplx)
>            ↓
>        joints[t,j,:]
>            ↓
>   pick joints for markers
>            ↓
>     loss(pick, markers)
> ```
>
> Because every step is differentiable, PyTorch can run `loss.backward()` and produce gradients for all trainable parameters.
> $$
> % Core idea: parameters -> loss
> \theta = (\text{global\_orient}, \text{body\_pose}, \text{transl}, \text{betas})
> \quad\Rightarrow\quad
> L(\theta) = \text{Loss}(\text{SMPL}(\theta), \text{markers})
> $$
> $$
> % Gradients tell how to change each parameter to decrease loss
> \nabla_{\theta} L
> =
> \left[
> \frac{\partial L}{\partial \text{global\_orient}},
> \frac{\partial L}{\partial \text{body\_pose}},
> \frac{\partial L}{\partial \text{transl}},
> \frac{\partial L}{\partial \text{betas}}
> \right]
> $$
> $$
> % Gradient descent style update (Adam is a fancier version of this)
> \theta \leftarrow \theta - \eta \, \nabla_{\theta} L
> $$

---

### Inputs

> [!info] I/O
> **Inputs:**
>
> * `smpl`: frozen `smplx` SMPL model
> * `markers_torch`: (T, K, 3) observed marker positions (**meters**)
> * `marker_joint_idx`: (K,) NumPy array, each element ∈ [0..23] (SMPL joint id)
> * `used_names`: list[str] length K (marker labels, aligned with `markers_torch[:,k,:]`)
> * `init_betas`: `None` or (10,) cached subject shape
> * `device`: `"cpu"` or `"cuda"`
> * `iters`: optimization iterations
> * weights: `w_marker, w_pose, w_betas, w_vel, w_acc`
>
> **Output:** dict with
>
> * `poses`: (T, 69) `body_pose` (axis-angle per joint)
> * `global_orient`: (T, 3) root orientation (axis-angle)
> * `trans`: (T, 3) translation
> * `betas`: (10,) shape
> * `joints`: (T, 24, 3) SMPL joints
> * `final_marker_loss`: float

> [!note] Constants vs per-frame
>
> * **Constant over T:** `betas`
> * **Per-frame:** `global_orient[t]`, `body_pose[t]`, `transl[t]`
>
> This matters because a bad shape can bias *every* frame, while a bad pose mainly affects local time.

---

### Data shapes at a glance

> [!tip] Shape cheat sheet
>
> * `markers_torch`: (T, K, 3)
> * `marker_joint_idx`: (K,)
> * `joints`: (T, 24, 3)
> * `pick = joints[:, targets_idx, :]`: (T, K, 3)
> * `body_pose`: (T, 69)
> * `global_orient`: (T, 3)
> * `transl`: (T, 3)

---

## Step-by-step

### 1) Create the parameter container

> [!example] Code
>
> ```python
> fitter = SMPLFitter(smpl, T, device, init_betas)
> ```
>
> > [!example]- What it does
> > * Allocates trainable tensors:
> > 	* `global_orient`: (T,3)
> > 	* `body_pose`: (T,69)
> > 	* `transl`: (T,3)
> > 	* `betas`: (10,)
> > * If `init_betas` is provided, shape starts close to the subject’s cached shape.

> [!note] Why this matters
> The SMPL model weights are frozen; only these parameters can move.

---

### 2) Initialize the hardest variables (root orientation + translation)

> [!example] Code
>
> ```python
> mk_np = markers_torch.detach().cpu().numpy()
> aa, pelvis_series = estimate_root_init(mk_np, used_names)
> fitter.global_orient.data[:] = torch.from_numpy(np.repeat(aa[None,:], T, axis=0))
> fitter.transl.data[:]        = torch.from_numpy(pelvis_series)
> ```

> [!example]- What it does
>
> * Sets **every frame’s** root orientation to the same `aa`
> * Sets translation to pelvis centroid per frame
>
>   * `transl[t] ≈ mean(LASI, RASI, SACR)` (if those exist)

> [!warning] Main failure mode
> If coordinate conventions are wrong (e.g., Vicon Z-up vs SMPL expectations), this initialization can be rotated/flipped.
> Then the optimizer spends iterations “fighting” the wrong root frame, which shows up as:
>
> * pelvis sideways
> * legs swapping front/back
> * leaning/drifting artifacts

---

### 3) Choose optimization speeds (learning rates)

> [!example] Code
>
> ```python
> opt = Adam([
>   {"params":[fitter.global_orient, fitter.body_pose, fitter.transl], "lr":3e-2},
>   {"params":[fitter.betas], "lr":1e-3}
> ])
> ```

> [!example]- Interpretation
>
> * Pose/orient/trans update **fast** (`3e-2`) because they must explain motion frame-by-frame
> * Shape updates **slow** (`1e-3`) because it’s shared across time and should not overfit jitter/noise

> [!note] Why this split exists
> Without smaller LR on `betas`, the fitter can “cheat”: distort body shape to match inconsistent markers rather than solving pose properly.

---

### 4) Define marker mismatch loss (robust)

> [!example] Code
>
> ```python
> hub = nn.SmoothL1Loss(beta=0.01, reduction='none')
> ```
>
> > [!example]- What it means
> > * This behaves like Huber / Smooth L1:
> >   * small errors ≈ L2 (stable gradients)
> >   * large errors ≈ L1 (outliers don’t dominate)

> [!note] Why `reduction='none'`
> You need per-element losses so you can apply custom marker weights (e.g., legs emphasized).

---

### 5) Build the joint-index tensor used for “gather”

> [!example] Code
>
> ```python
> targets_idx = torch.as_tensor(marker_joint_idx, device=device, dtype=torch.long)
> ```
>
> > [!example]- Meaning
> > For each marker `k`, you will compare it to SMPL joint `targets_idx[k]`.

> [!note] Mapping reminder
> `marker_joint_idx` comes from `subset_markers`, so the mapping is:
>
> * marker track `markers[:,k,:]`
> * ↔ SMPL joint `joints[:, targets_idx[k], :]`

---

## The training loop (one iteration)

### 6.1 Forward pass: parameters → joints → marker-matched joints

> [!example] Code
>
> ```python
> joints = fitter()               # (T,24,3)
> pick  = joints[:, targets_idx]  # (T,K,3)
> ```
>
> [!example]- Interpretation
>
> * `joints[t,j,:]` is SMPL joint j position at time t
> * `pick[t,k,:]` is the predicted joint position associated with marker k

> [!warning] A subtle limitation
> The script is matching *markers* to *joints*, but many markers are not actually located at joint centers (ASIS/heel/toe/head markers).
> That mismatch can bias pose unless you add:
>
> * offsets, or
> * multi-marker rigid segment constraints, or
> * better marker-to-surface correspondence (MoSh-like)

---

### 6.2 Build per-marker weights (leg emphasis)

> [!example] Code
>
> ```python
> w = np.ones(K, dtype=np.float32)
> leg_keys = {"LKNE","RKNE","LANK","RANK","LHEE","RHEE","LTOE","RTOE"}
> w[[i for i,n in enumerate(names) if n in leg_keys]] = 1.5
> w_t = torch.from_numpy(w).to(device)[None,:,None]  # (1,K,1)
> ```
>
> > [!example]- Meaning
> > * Leg markers get 1.5× higher weight, so legs dominate the fit more than upper body.
> > * This helps stabilize gait (knee/ankle placement), but can also:
> > * cause torso to “sacrifice” accuracy if weights are too high.

---

### 6.3 Marker loss (robust) + weighting

> [!example] Code
>
> ```python
> per_elem = hub(pick, markers_torch)      # (T,K,3)
> marker_loss = (per_elem * w_t).mean()
> ```
>
> > [!example]- Interpretation
> > * Compute SmoothL1 per coordinate, per marker, per frame
> > * Multiply by weight per marker
> > * Average everything into one scalar

> [!note] What this scalar represents
> Roughly: “average weighted marker error in meters”, but not exactly L2 distance because SmoothL1 mixes L1/L2.

---

### 6.4 Priors and smoothness terms

> [!summary] Why these exist
> Marker fitting alone is underconstrained: the model can twist in implausible ways to chase marker noise.
> Priors stop the optimizer from “doing something mathematically valid but anatomically cursed.”

#### Pose L2 prior

> [!example] Code
>
> ```python
> pose_l2 = (fitter.body_pose**2).mean()
> ```
>
> > [!example]- Meaning
> > Penalizes large joint rotations across all frames.
> > Acts like a weak “stay near zero pose” prior.

#### Betas L2 prior

> [!example] Code
>
> ```python
> betas_l2 = (fitter.betas**2).mean()
> ```
>
> > [!example]- Meaning
> >Keeps body shape near the mean human (prevents extreme shapes).

#### Translation velocity smoothness

> [!example] Code
>
> ```python
> vel = (transl[1:] - transl[:-1]).pow(2).mean()
> ```
>
> > [!example]- Meaning
> > Penalizes frame-to-frame translation jitter.

#### Translation acceleration smoothness

> [!example] Code
>
> ```python
> acc = (transl[2:] - 2*transl[1:-1] + transl[:-2]).pow(2).mean()
> ```
>
> > [!example]- Meaning
> > Penalizes sudden changes in velocity (makes transl trajectory “smoothly accelerating”)

> [!warning] Missing smoothness
> There is **no temporal smoothness on `body_pose` or `global_orient`** in this old fitter.
> So you can still get:
>
> * spine jiggle
> * single-frame pose spikes
> * inconsistent torso twist
>   even if translation is smooth.

---

### 6.5 Total loss

> [!example] Code
>
> ```python
> loss = (
>   w_marker*marker_loss +
>   w_pose*pose_l2 +
>   w_betas*betas_l2 +
>   w_vel*vel +
>   w_acc*acc
> )
> ```
>
> [!note] Interpretation
> This is a weighted sum of five competing goals:
>
> * match markers
> * keep pose small-ish
> * keep shape reasonable
> * keep translation smooth (1st derivative)
> * keep translation smooth (2nd derivative)

---

### 6.6 Backprop + parameter update

> [!example] Code
>
> ```python
> loss.backward()
> opt.step()
> ```
>
> > [!example]- Meaning
> > * Backprop computes gradients w.r.t. `global_orient`, `body_pose`, `transl`, `betas`
> > * Adam updates them using their respective learning rates

> [!note] Logging
> Prints loss every 100 iterations to monitor convergence.

---

## Final output packing

> [!summary] What happens after optimization
>
> * Run `joints = fitter()` one last time with final parameters
> * Return NumPy arrays for saving

> [!info] Output dict fields
>
> * `poses`: body_pose (T,69)
> * `global_orient`: (T,3)
> * `trans`: (T,3)
> * `betas`: (10,)
> * `joints`: (T,24,3)
> * `final_marker_loss`: scalar

---

## `main()`

> [!summary] Purpose
> Orchestrate the entire batch fitting job:
>
> * iterate over subjects (or one subject)
> * optionally filter to a specific trial
> * ensure each subject has a cached **shape** (`betas.npy`)
> * fit **pose + translation** for every trial using that cached shape
> * write per-trial outputs (`*_smpl_params.npz`) + a lightweight JSON report

> [!abstract] High-level pipeline
>
> ```text
> processed_dir/
>   SUBJ01/
>     SUBJ1_..._markers_positions.npz
>     SUBJ1_..._metadata.json
>   SUBJ02/
>     ...
>
> main():
>   for each subject:
>     load metadata → pick gender → build SMPL model
>     ensure betas.npy exists (fit once if missing)
>     for each trial:
>       load markers → subset → prep → torch
>       fit_sequence(...) → save NPZ + JSON report
> ```

---

## CLI arguments

> [!info] Inputs you control
>
> * `--processed_dir`
>   Root directory containing one folder per subject (e.g. `processed_dir/SUBJ01/`)
> * `--models_dir`
>   SMPL model directory passed into `smplx.create(...)`
> * `--out_dir`
>   Output root; script creates `out_dir/SUBJXX/` per subject
> * `--subject` *(optional)*
>   If provided, only process that one subject folder
> * `--trial` *(optional)*
>   Filter trials by filename prefix (based on `Path(npz).stem.startswith(args.trial)`)
> * `--device`
>   `"cpu"` or `"cuda"` (CUDA only if `torch.cuda.is_available()`)
> * `--iters`
>   Default 400; used for per-trial fitting iterations
>
>   * beta-fitting phase uses `min(iters*2, 800)` in this old script

> [!tip] Practical note on `--trial`
> It matches by **stem prefix**, so if your trial names contain spaces/parentheses, make sure you pass the prefix exactly as it appears in the stem *before* sanitization.

---

# Subject discovery and loop

## 1) Discover subjects

> [!summary] Purpose
> Decide which subject folders to process.

> [!example]- Behavior
>
> * If `--subject` is provided:
>
>   * `subjects = [args.subject]`
> * Else:
>
>   * `subjects = sorted(all directories under processed_dir)`

> [!note] Assumption
> Every directory under `processed_dir/` is treated as a subject folder.
> If you have other directories in there, they’ll be included unless you filter.

---

## 2) Per-subject setup

For each subject folder `processed_dir/SUBJXX/`:

> [!example]- What main() gathers
>
> 1. **All trials**
>
>    * Find `*_markers_positions.npz` in that subject directory
>    * If none found → skip subject
> 2. **Metadata (gender)**
>
>    * Find any `*_metadata.json`
>    * Read the first one and run `pick_gender(meta)`
>    * Default fallback is `"neutral"`
> 3. **Output paths**
>
>    * Create `out_dir/SUBJXX/`
>    * Define `betas_path = out_dir/SUBJXX/betas.npy`
> 4. **SMPL model**
>
>    * Build once per subject:
>
>      ```python
>      smpl = build_smpl(models_dir, gender, device)
>      ```
>    * Frozen model; reused for all that subject’s trials

> [!note] Efficiency win
> SMPL is built **once per subject**, not per trial.
> That’s correct because only the parameters change across trials.

---

# Betas caching logic (per subject)

## 3) Shape estimation is separated from motion estimation

> [!summary] Why this exists
> SMPL shape (`betas`) is assumed **constant for a subject**.
> So the script tries to:
>
> * estimate `betas` once
> * reuse it across trials
>   so later trials don’t “re-invent” body proportions each time.

### A) If `betas.npy` exists

> [!example] Behavior
>
> * Load cached betas:
>
>   ```python
>   init_betas = np.load(betas_path)  # (10,)
>   ```
> * All trials start from this subject-specific body shape.

> [!note] Impact on stability
> This typically reduces:
>
> * per-trial variance
> * weird limb length drift
> * “overfitting shape to noisy markers”

### B) If `betas.npy` is missing

> [!example]- One-time “beta fitting” phase
> The script picks **the first trial file** and runs a shortened fit intended primarily to get shape:
>
> 1. Load the first trial’s markers:
>
>    * `M, names, fps = load_markers_npz(first_npz)`
> 2. Subset markers to the anatomical set:
>
>    * `M_sub, used_names, tgt_idx = subset_markers(M, names)`
> 3. Drop bad frames + fill NaNs:
>
>    * `M_sub, keep = prep_markers(M_sub, ...)`
> 4. Optionally downsample to speed up:
>
>    * if `T > 220`, decimate frames to about 20 fps and cap around ~220 frames
>      (this makes “shape estimation” faster)
> 5. Convert to torch:
>
>    * `markers_torch = torch.from_numpy(M_sub).float().to(device)`
> 6. Fit with `init_betas=None`:
>
>    * `fit_sequence(..., init_betas=None, iters=min(iters*2, 800))`
> 7. Save:
>
>    * `np.save(betas_path, result["betas"])`

> [!warning] Design limitation (important)
> This “fit betas on one trial” assumes:
>
> * that first trial is representative and clean
> * markers are well-behaved (standing trial often is better; but not guaranteed)
>
> If the chosen trial has missing/rotated markers, the cached betas can be biased and will affect every later trial.

> [!tip] If you want to improve the pipeline later
> A stronger variant is “fit betas using multiple trials (or the standing trial), then lock betas.”

---

# Trial fitting loop

## 4) Fit each trial using cached betas

For every file `*_markers_positions.npz` in the subject folder:

> [!summary] Purpose
> Estimate per-frame motion parameters (pose/orient/trans) given a fixed subject shape.

> [!example]- Steps
>
> 1. Construct output filenames:
>
>    * `trial_name` = stem without `_markers_positions`
>    * `trial_safe = sanitize(trial_name)` (used only for filenames)
>    * `out_trial = out_dir/SUBJXX/{trial_safe}_smpl_params.npz`
>    * `report_path = out_dir/SUBJXX/{trial_safe}_smpl_metadata.json`
> 2. Skip if already fitted:
>
>    * If `out_trial.exists()` → continue
> 3. Load + preprocess markers:
>
>    * `M, names, fps = load_markers_npz(npz_path)`
>    * `M_sub, used_names, tgt_idx = subset_markers(M, names)`
>    * `M_sub, keep = prep_markers(M_sub, ...)`
>    * `markers_torch = torch.from_numpy(M_sub).float().to(device)`
> 4. Fit:
>
>    * `result = fit_sequence(..., init_betas=cached_betas, iters=args.iters)`
> 5. Save outputs:
>
>    * NPZ with SMPL params + joints
>    * JSON report with summary stats

---

## Output: NPZ format

> [!info] What gets saved (per trial)
> The script writes:
>
> * `poses`: (T,72) where 72 = 3 (root) + 69 (body)
> * `trans`: (T,3)
> * `betas`: (10,)
> * `joints`: (T,24,3)
> * plus metadata fields like `gender`, `subject_id`, `trial_name`, `fps`, `n_frames`

> [!example]- Code path
>
> ```python
> poses72 = np.concatenate([result["global_orient"], result["poses"]], axis=1)  # (T,72)
> np.savez(out_trial,
>   poses=poses72,
>   trans=result["trans"],
>   betas=result["betas"],
>   gender=gender,
>   subject_id=subj,
>   trial_name=trial_name,
>   fps=fps,
>   n_frames=result["trans"].shape[0],
>   joints=result["joints"]
> )
> ```

> [!note] Why store `joints` too?
> It’s redundant (you could recompute joints from SMPL params), but it makes debugging and downstream export easier.

---

## Output: JSON report

> [!summary] Purpose
> Lightweight record of “what happened” during fitting so you can audit results without loading huge arrays.

> [!info] Typical fields
>
> * `subject_id`, `trial_name`, `gender`
> * `frames_kept` (after dropping bad frames)
> * `fps`
> * `markers_used` (the final K marker names after subsetting)
> * `final_marker_loss` (final smoothL1 marker mismatch scalar)

> [!tip] Why this report matters
> When a trial looks bad, you can quickly check:
>
> * did it keep very few frames?
> * did it use too few markers?
> * did loss remain high compared to other trials?

---

# End-to-end data flow (shapes-first)

> [!abstract] One trial, fully expanded
>
> 1. **Load NPZ**
>
> * `M`: (T, N, 3) meters, NaNs for missing
>
> * `names`: (N,)
>
> * `fps`: scalar
>
> 2. **Subset anatomical markers**
>
> * `M_sub`: (T, K, 3)
>
> * `used_names`: (K,)
>
> * `tgt_idx`: (K,) → SMPL joint ids
>
> 3. **Prep**
>
> * `keep_frames`: (T,) bool
>
> * `M2`: (T_kept, K, 3) (frames dropped; NaNs interpolated)
>
> 4. **Torch conversion**
>
> * `markers_torch`: (T_kept, K, 3) float32
>
> 5. **Fit** (`fit_sequence`)
>
> * initialize `global_orient[t] = aa` for all t
> * initialize `transl[t] = pelvis_centroid(t)`
> * optimize marker loss + priors + translation smoothness
> * output:
>
>   * `global_orient`: (T_kept, 3)
>
>   * `body_pose`: (T_kept, 69)
>
>   * `trans`: (T_kept, 3)
>
>   * `betas`: (10,)
>
>   * `joints`: (T_kept, 24, 3)
>
> 6. **Save**
>
> * `poses72`: (T_kept,72) stored as `poses`
> * plus `trans`, `betas`, `joints`, and metadata

---

# What this old orchestration *implicitly assumes*

> [!warning] Why main() “worked” but results failed
> The driver is mechanically correct, but it bakes in assumptions that can break fitting:

1. **Coordinate systems already match**

* No global conversion in `main()`
* So if your dataset is Z-up and SMPL expects something else, every trial inherits that mismatch.

2. **The “first trial” is good enough to estimate body shape**

* If first trial has noise / marker swaps / missing data, `betas.npy` becomes contaminated.

3. **Per-trial fitting is independent (no temporal warm-start across trials)**

* Each trial starts from the same cached betas, but pose is reset and solved fresh.
* That’s fine, but doesn’t exploit easy wins like reusing a good root yaw estimate per subject.

4. **Quality control is minimal**

* It records marker loss, but doesn’t automatically reject bad trials or refit them differently.
