# Bug Report: Canonical Axis Inversion in `3_export_humanml3d.py`

**Date:** 2026-03-27  
**Fixed in commit:** `c93d7ae8b3fa14f067f99fe8c8be3de244b4f6f5`  
**Buggy commit:** `91fd2a7ff5aa5e0d9bb1a60d379ab9eee27134ba` (and all prior)

---

## Summary

A critical error in `rot_to_align()` inside `3_export_humanml3d.py` caused the canonical X-axis to be **inverted**, producing a left-right mirror of every exported motion clip. The body's anatomical left side was placed at canonical −X instead of the required +X. The canonical Y and Z axes were unaffected — the forward direction was computed correctly in both versions.

The bug was confirmed numerically (all left-side joints have negative X in buggy data, positive X in fixed data) and visually: fitting an SMPL mesh to the buggy joints via `visualize_joints/fit_seq.py` and rendering with `visualize_joints/render.py` produced a moonwalk-like artefact, while the fixed version produces normal forward gait. The moonwalk arises because SMPLify-3D, when fitted to X-mirrored joint targets, converges to body_pose parameters where the knee flexion and foot articulation are anatomically inverted — causing the limbs to appear to push in the wrong direction even though the root translation moves forward.

---

## The Bug

### Root cause

The single root cause is in `rot_to_align()`: it discards the `left` vector measured directly from joint positions and re-derives a left-axis from `cross(fwd, up)`. For the actual Van Criekinge data (subject walking in world +X direction), this cross product yields world +Z — which is the body's **right** side, not left. This caused the canonical +X column of the rotation matrix to point toward body-right throughout.

The `estimate_axes()` function also had a structural issue (computing `right` instead of `left`, and returning a redundant `lr` that `rot_to_align` never used), but these did not independently cause the error: the forward direction computed by `estimate_axes` was actually correct for the VC walking direction. The failure was entirely in `rot_to_align` throwing away the measured lateral axis and re-deriving the wrong one.

### Incorrect code (HEAD~1, `91fd2a7`)

```python
def estimate_axes(j):
    up = normalize(j[:, 12, :] - j[:, 0, :])   # pelvis → neck  ✓

    # Computes body-RIGHT (R_hip − L_hip), not body-left
    right = normalize(j[:, 2, :] - j[:, 1, :])  # R_hip − L_hip = rightward

    fwd = normalize(np.cross(up, right))

    # lr is computed and returned but rot_to_align never uses it
    lr = normalize(np.cross(up, fwd))

    return up, fwd, lr


def rot_to_align(up, fwd):          # lr not accepted — silently discarded by caller
    u = normalize(up.mean(axis=0), ...).squeeze()
    f = normalize(fwd.mean(axis=0), ...).squeeze()

    # ROOT CAUSE: re-derives left from scratch.
    # cross(fwd, up) = body-RIGHT for any walking direction, not body-left.
    l = normalize(np.cross(f, u), ...).squeeze()

    R_src = np.stack([l, u, f], axis=1)
    return R_src.T
```

### Fixed code (HEAD, `c93d7ae`)

```python
def estimate_axes(j):
    up = normalize(j[:, 12, :] - j[:, 0, :])   # pelvis → neck  ✓

    # L_hip − R_hip correctly gives the body-left direction
    left = normalize(j[:, 1, :] - j[:, 2, :])  # joints: 1=L_hip, 2=R_hip

    fwd = normalize(np.cross(left, up))

    return up, fwd, left   # left is passed through to rot_to_align


def rot_to_align(up, fwd, left):
    u = normalize(up.mean(axis=0), ...).squeeze()
    f = normalize(fwd.mean(axis=0), ...).squeeze()

    # FIX: use the directly measured left vector instead of re-deriving
    l = normalize(left.mean(axis=0), ...).squeeze()

    R_src = np.stack([l, u, f], axis=1)   # columns = [left, up, fwd]
    return R_src.T                          # world → canonical
```

---

## Numerical Confirmation

Comparing `data/humanml3d_joints_4/` (HEAD~1) vs `data/humanml3d_joints_TEST/` (HEAD) on subject SUBJ01, trial 1:

| Joint | HEAD~1 mean X | HEAD mean X | Expected sign |
|-------|-------------|-------------|---------------|
| L_hip (idx 1) | −0.054 | **+0.050** | **positive** |
| R_hip (idx 2) | +0.058 | **−0.062** | **negative** |
| L_shoulder (idx 16) | −0.205 | **+0.225** | **positive** |
| R_shoulder (idx 17) | +0.183 | **−0.163** | **negative** |

Pelvis trajectory (canonical Z and X):

| Version | Z displacement | X displacement |
|---------|---------------|----------------|
| HEAD~1 | +3.54 m | +0.245 m |
| HEAD   | +3.54 m ✓ | −0.241 m |

The Z displacement is **identical** in both versions — confirming the bug was a pure X-axis inversion with no effect on canonical Y or Z. The rotation matrices for both versions share identical rows 1 and 2 (canonical Y and Z), differing only in row 0 (canonical X), which is sign-flipped. Max absolute joint position difference: **0.628 m**.

---

## Justification: Official HumanML3D Convention

The canonical frame required by HumanML3D is justified solely by reference to the **official HumanML3D repository** (`EricGuo5513/HumanML3D`, `raw_pose_processing.ipynb`).

### Evidence 1: X-axis negation in the processing loop

Cell 13 of `raw_pose_processing.ipynb`:

```python
for i in tqdm(range(total_amount)):
    ...
    if 'humanact12' not in source_path:
        ...
        data[..., 0] *= -1      # ← explicit X-axis negation applied to ALL AMASS clips

    data_m = swap_left_right(data)
    np.save(pjoin(save_dir, new_name), data)
    np.save(pjoin(save_dir, 'M'+new_name), data_m)
```

AMASS SMPL joint positions — after the `trans_matrix` step that converts from mocap Z-up to Y-up — are in a standard right-handed frame where +X = body-right. The explicit `data[..., 0] *= -1` flips this, making **+X = body-left** the canonical convention for all stored joints.

HumanAct12 data (skeleton from RGB-D/Kinect, where the sensor's +X naturally corresponds to the subject's left) is exempt from this flip — it is the reference convention that AMASS data is adjusted to match.

Van Criekinge data, like AMASS, is captured in a Vicon Z-up system and converted to Y-up by the SMPL fitting pipeline. It therefore starts in the same right-handed X-right state as AMASS after `trans_matrix`, and requires the equivalent X-flip that `rot_to_align` is responsible for producing.

### Evidence 2: `swap_left_right` defines the left/right chain unambiguously

Cell 11 of `raw_pose_processing.ipynb`:

```python
def swap_left_right(data):
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]   # R_hip, R_knee, R_ankle, ...
    left_chain  = [1, 4, 7, 10, 13, 16, 18, 20]   # L_hip, L_knee, L_ankle, ...
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    return data
```

`swap_left_right` is the mirroring function used for data augmentation. It creates a physically valid mirror by (a) negating X and (b) swapping the left and right joint index groups. The fact that joint **index 1 belongs to the left chain** and joint **index 2 belongs to the right chain** is the authoritative statement of joint ordering from the official codebase.

Therefore:
- Joint 1 = L_hip → must sit at **positive X** (canonical +X = body-left)
- Joint 2 = R_hip → must sit at **negative X**

The HEAD~1 (buggy) export places L_hip at negative X, violating this. The HEAD (fixed) export places L_hip at positive X, conforming to it.

---

## Visual Confirmation

After running `visualize_joints/fit_seq.py` (SMPLify-3D fitting of an SMPL mesh onto the exported 22-joint skeletons) and viewing the result with `visualize_joints/render.py`:

- **HEAD~1 (buggy):** subject performs a moonwalk — the root translates forward but the limb articulation appears to push backward. This arises because SMPLify-3D fitted to X-mirrored joint targets converges to body_pose values where knee flexion and foot strike are anatomically inverted relative to the direction of travel.
- **HEAD (fixed):** subject walks normally, with feet and body translation consistently advancing in the forward direction.

Note: the moonwalk artefact is **not** caused by Z-axis inversion (the canonical Z and the pelvis trajectory are identical in both versions). It is purely a consequence of left-right mirrored joint positions producing anatomically inconsistent SMPL poses.

---

## Impact Assessment

| Downstream step | Impact of bug |
|---|---|
| Step 4 feature extraction (263-dim) | Joint rotations computed with swapped L/R limbs — all limb-specific features wrong |
| Foot contact labels | Left/right foot contact swapped throughout |
| ST-GCN++ age classifier | Gait is largely bilateral; age signal mostly preserved, but laterality corrupted |
| Text-conditioned LoRA-MDM | "Left arm" text would generate right arm motion and vice versa |
| Data augmentation (`swap_left_right`) | Augmented mirrors of buggy data are also consistently wrong — augmentation does not self-correct the bug |
