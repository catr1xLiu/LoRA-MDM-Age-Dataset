>[!info|right]
> **Part of a series on**
> ###### [[Dataset Overview|Van Criekinge]]
> ---
> **Pipeline**
> * [[1 VC Pipeline/Pipeline Overview|Pipeline Overview]]
> * [[1 VC Pipeline/dataset_prep.py fixes|Dataset Prep]]
> * [[1 VC Pipeline/motion_process.py fixes|Motion Process]]

>[!info]
> ###### Motion Process Fix
> [Type::Code Maintenence]
> [Script::motion_process.py]
> [Issue::Zero Velocity (Moonwalking)]
> [Status::Resolved]

This document addresses a critical bug in `motion_process.py` where root velocity was incorrectly calculated as zero, leading to "moonwalking" artifacts in generated motion (stationary characters gliding).


## Problem Description
During the processing of the Van Criekinge dataset, it was discovered that `Std.npy` contained zero values at indices 1 and 2 (root planar linear velocity). This caused the model to hallucinate stationary movement, as the training data suggested the character was moving in place even during significant travel.

## Root Cause
The zero velocities occurred because `export_humanml3d.py` canonicalizes sequences by centering joints at (0,0,0). While it saves the global trajectory in a separate `pelvis_traj` array, the original `motion_process.py` only read the `joints` array, effectively ignoring travel information.


## Implemented Solution
The solution modifies the script to extract the `pelvis_traj` array provided by the export script and use it to calculate the correct root velocity.

### 1. `build_vc_dataset` Updates
Modified the main loop to load `pelvis_traj` from the NPZ and pass it to feature processing.


```python
# Inside build_vc_dataset's loop:
d = np.load(fin, allow_pickle=True)
joints = _load_joints(d).astype(np.float32)

# ADDED:
pelvis_traj_data = d['pelvis_traj'].astype(np.float32) if 'pelvis_traj' in d.files else None

# MODIFIED:
feats, _, _, _ = process_file(joints, 0.002, pelvis_traj=pelvis_traj_data)
```

### 2. Signature Update
Updated `process_file` to accept the new optional argument.
```python
def process_file(positions, feet_thre, pelvis_traj=None): # Updated signature
```

### 3. Trajectory Alignment
Inserted logic to align the pelvis trajectory with the joint rotation.
*   **Zero Origin**: Aligns initial XZ to zero.
*   **Rotate**: Applies the same `root_quat_init` used on the joints.


```python
# ADDED (around line 310):
# --- Align pelvis trajectory to the same origin & facing (if provided) ---
pel_aligned = None
if pelvis_traj is not None:
    pel_aligned = pelvis_traj.astype(np.float32).copy()
    # zero initial XZ so sequences are comparable to positions processing
    pel_aligned[:, [0, 2]] -= pel_aligned[0, [0, 2]]
    # rotate with the same initial facing quaternion used on joints
    # root_quat_init has shape (T, J, 4); take the root joint channel
    pel_aligned = qrot_np(root_quat_init[:, 0, :], pel_aligned)  # (T, 3)
```

### 4. Velocity Calculation
Replaced the direct velocity extraction with conditional logic.
*   **If Trajectory Exists**: Calculate velocity from `pel_aligned` (matches world traversal).
*   **Fallback**: Use original joint velocity (zero for root-centered data).

```python
# MODIFIED (around line 396):
# Compute planar root linear velocity from pelvis trajectory if available
if pel_aligned is not None:
    # world-space Δ position per frame
    vel_world = pel_aligned[1:] - pel_aligned[:-1]       # (T-1, 3)
    # rotate into the root frame of the current sequence
    vel_local = qrot_np(r_rot[1:], vel_world)            # (T-1, 3)
    l_velocity = vel_local[:, [0, 2]]                    # XZ components
else:
    # fallback: infer from positions (will be ~0 if joints are root-centered)
    l_velocity = velocity[:, [0, 2]]

root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)
```

### 5. Main Block Updates
Mirrored the changes in the `if __name__ == "__main__":` block to support single-file testing.

---

## Results
Implementation confirmed correct non-zero averages for indices 1 and 2 in `Std.npy`, representing actual traversal speed, while preserving zero local root velocity (indices 193-195).