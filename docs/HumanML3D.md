# HumanML3D Coordinate System & Joint Naming Conventions

## Coordinate System

HumanML3D uses a **Y-up, right-handed** coordinate system with the **XZ plane as the ground**.

| Axis | Direction |
|------|-----------|
| **+X** | Body-left (anatomical) |
| **+Y** | Up |
| **+Z** | Forward (facing direction) |

This is right-handed because `cross(+X, +Y) = +Z`, i.e. `cross(left, up) = forward`.

### How to visualize it

Take a standard X-Y plane on paper (+X right, +Y up). Now imagine the paper is your **screen** instead of the floor. The third axis, Z, comes out of the screen toward you. But HumanML3D has +Z as **forward** (into the scene), so the camera looks down the **-Z** axis when viewing the character's face.

## The 263-Dimensional Feature Vector

The per-frame motion representation (applied after all preprocessing) encodes the coordinate convention directly:

- **Root angular velocity** (1D): a single scalar for yaw rotation around the **Y axis** only
- **Root linear velocity** (2D): velocity on the **XZ ground plane** (indices 0 and 2 of the 3D position)
- **Root height** (1D): the **Y** coordinate of the pelvis
- **Local joint positions** (63D): relative to root, in the root-aligned frame
- **Joint rotations** (126D): continuous 6D rotation representation for 21 non-root joints
- **Joint velocities** (66D): local frame velocities for all 22 joints
- **Foot contacts** (4D): binary contact labels for left and right feet

## Joint Naming: The X-Negation Problem

### What happened

The HumanML3D preprocessing pipeline (`raw_pose_processing`) applies an X-axis negation to all AMASS-sourced data:

```python
data[..., 0] *= -1   # in the segmentation loop, applied to all non-humanact12 data
```

This mirrors the entire skeleton across the YZ plane **after** SMPL assigned joint names based on anatomy. The joint indices and their "L\_" / "R\_" labels were never updated to reflect the mirror.

### The result

**In the final HumanML3D data, joint names are swapped relative to anatomy:**

| Joint name | Joint index | X position | Anatomical side |
|-----------|-------------|------------|-----------------|
| `L_hip` | 1 | **+X** | Character's **right** |
| `R_hip` | 2 | **-X** | Character's **left** |
| `L_shoulder` | 16 | **+X** | Character's **right** |
| `R_shoulder` | 17 | **-X** | Character's **left** |

In other words: "L\_" joints are on the character's anatomical **right** side, and "R\_" joints are on the character's anatomical **left** side.

### Why it still works

The pipeline is internally consistent. The `swap_left_right` mirror function used for data augmentation accounts for this by negating X again and swapping left/right chain indices:

```python
def swap_left_right(data):
    data[..., 0] *= -1          # mirror X
    # then swap joint indices between left_chain and right_chain
```

The facing direction computation in `process_file` also uses the post-negation positions consistently, so the canonical frame comes out correct.

### Camera view intuition

When the camera faces the character's face (looking down -Z):

- "L\_" joints appear on the **left side of the screen** (positive X goes left on screen)
- "R\_" joints appear on the **right side of the screen**

This is the "camera left/right" convention, not the anatomical one. It matches what you see in a mirror.

## The 22 SMPL Joints

```
Index  Name           Chain     Post-negation X side
─────  ────────────   ────────  ────────────────────
  0    pelvis         center    ~0
  1    L_hip          left      +X (anatomical RIGHT)
  2    R_hip          right     -X (anatomical LEFT)
  3    spine1         center    ~0
  4    L_knee         left      +X (anatomical RIGHT)
  5    R_knee         right     -X (anatomical LEFT)
  6    spine2         center    ~0
  7    L_ankle        left      +X (anatomical RIGHT)
  8    R_ankle        right     -X (anatomical LEFT)
  9    spine3         center    ~0
 10    L_foot         left      +X (anatomical RIGHT)
 11    R_foot         right     -X (anatomical LEFT)
 12    neck           center    ~0
 13    L_collar       left      +X (anatomical RIGHT)
 14    R_collar       right     -X (anatomical LEFT)
 15    head           center    ~0
 16    L_shoulder     left      +X (anatomical RIGHT)
 17    R_shoulder     right     -X (anatomical LEFT)
 18    L_elbow        left      +X (anatomical RIGHT)
 19    R_elbow        right     -X (anatomical LEFT)
 20    L_wrist        left      +X (anatomical RIGHT)
 21    R_wrist        right     -X (anatomical LEFT)
```

Joints 22 and 23 (hand tips) exist in SMPL-24 but are dropped by HumanML3D, which uses only the first 22.

## Preprocessing Pipeline Summary

The full pipeline from raw AMASS to final HumanML3D representation:

1. **SMPL forward kinematics** → 3D joint positions from body parameters (Z-up in SMPL)
2. **Y-Z axis swap** → converts from Z-up to Y-up via `trans_matrix = [[1,0,0],[0,0,1],[0,1,0]]`
3. **X-negation** → `data[..., 0] *= -1` flips handedness (and swaps L/R naming)
4. **Temporal segmentation** → clip to annotated start/end frames
5. **Skeleton normalization** → retarget to a uniform skeleton via IK/FK
6. **Floor alignment** → subtract minimum Y so feet touch Y=0
7. **XZ centering** → first frame pelvis at XZ origin
8. **Face Z+** → rotate around Y axis so initial facing direction aligns with +Z
9. **Feature extraction** → root-centering each frame, computing the 263D vector (velocities, rotations, positions, foot contacts)
10. **Downsample to 20 FPS**

## References

- [HumanML3D GitHub](https://github.com/EricGuo5513/HumanML3D) — official dataset and processing scripts
- `raw_pose_processing.ipynb` — AMASS extraction, X-negation, segmentation, mirroring
- `motion_representation.ipynb` — `process_file`, 263D feature extraction, recovery functions
- Guo et al., "Generating Diverse and Natural 3D Human Motions from Text" (CVPR 2022)
