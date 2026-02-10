# Van Criekinge Dataset Marker Analysis Report

This document details the marker configurations used across different stages of the Van Criekinge motion processing pipeline, comparing the original raw dataset markers with the subsets used in previous and current SMPL fitting implementations.

## 1. Raw Dataset Markers (Van Criekinge / Plug-In Gait)

The raw dataset utilizes a full-body marker set consisting of **111 position markers**. This includes the standard 28 Plug-In Gait (PiG) markers plus additional tracking clusters (4 markers per segment) to ensure robustness during dynamic movements.

### Primary Anatomical Markers (Plug-In Gait)
- **Head:** `LFHD`, `RFHD`, `LBHD`, `RBHD`
- **Torso:** `C7`, `T10`, `CLAV`, `STRN`
- **Upper Limbs:** `LSHO`/`RSHO`, `LELB`/`RELB`, `LWRA`/`LWRB`, `RWRA`/`RWRB`, `LFIN`/`RFIN`
- **Pelvis:** `LASI`, `RASI`, `SACR`
- **Lower Limbs:** `LTHI`/`RTHI`, `LKNE`/`RKNE`, `LTIB`/`RTIB`, `LANK`/`RANK`, `LHEE`/`RHEE`, `LTOE`/`RTOE`, `LMT5`/`RMT5`

### Segment Clusters (Robust Tracking)
Each segment (Humerus, Radius, Femur, Tibia, etc.) features a 4-marker cluster (O, A, L, P suffixes) used to define local coordinate systems and track rigid body transformations.

---

## 2. Previous Fitting Implementation (`2_fit_smpl_markers.py`)

The original fitting script used a direct **Marker-to-Joint** mapping. It relied on a hardcoded subset of **29 markers** to estimate SMPL joint positions.

| SMPL Joint | Van Criekinge Markers |
| :--- | :--- |
| **Head** | `LFHD`, `RFHD`, `LBHD`, `RBHD` |
| **Spine/Neck** | `C7`, `T10`, `CLAV`, `STRN` |
| **Pelvis** | `LASI`, `RASI`, `SACR` |
| **Shoulder** | `LSHO`, `RSHO` |
| **Elbow** | `LELB`, `RELB` |
| **Wrist** | `LWRA`, `LWRB`, `RWRA`, `RWRB` |
| **Knee** | `LKNE`, `RKNE` |
| **Ankle** | `LANK`, `RANK`, `LHEE`, `RHEE` |
| **Foot** | `LTOE`, `RTOE`, `LMT5`, `RMT5` |

**Limitations:** This method ignored the surface geometry of the SMPL model and relied on joint center estimation, which often led to "sliding" markers or inaccurate limb orientations.

---

## 3. Current Fitting Implementation (`fit_markers.py`)

The current pipeline uses the `human_body_prior` library's `IK_Engine`. This version maps markers to **specific SMPL mesh vertex IDs**, allowing for a high-fidelity "surface fitting" optimization.

### Mapping Logic
The system uses a two-step mapping:
1. **Labels Map (`labels_map.py`):** Converts dataset-specific names to canonical AMASS/Mosh labels (e.g., `LASI` -> `LFWT`).
2. **Vertex IDs (`marker_vids.py`):** Maps canonical labels to vertex indices on the SMPL mesh (e.g., `LFWT` -> Vertex `857`).

### Markers Utilized (~35 Markers)
This version utilizes a more sophisticated subset that captures surface orientation:
- **Pelvis:** `LASI`/`RASI` mapped to Front Waist (`LFWT`/`RFWT`) vertices.
- **Wrists:** `LWRA`/`LWRB` mapped to Inner/Outer Wrist (`LIWR`/`LOWR`) vertices, providing better hand rotation.
- **Feet:** `LHEE`, `LTOE`, and `LMT5` are all mapped to their respective mesh locations, ensuring the foot model aligns with the ground plane and heel/toe markers simultaneously.
- **Fingers:** `LFIN`/`RFIN` are included to constrain the hand pose.

**Advantages:** By fitting to the mesh surface (with a 0.0095m marker offset), the `IK_Engine` produces much more anatomically correct results and utilizes the VPoser prior to prevent unnatural joint angles.
