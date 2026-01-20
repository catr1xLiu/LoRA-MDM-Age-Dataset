# VC Pipeline v2 (lora_mdm_age_dataset_2)

**Status:** Active | **Role:** Production VC Processing | **Type:** Repository Documentation

This repository contains the **corrected and production-ready** Van Criekinge processing pipeline. It supersedes the v1 pipeline (`lora_mdm_age_dataset`) with fixed SMPL marker fitting.

## Overview

This is part of the LoRA-MDM motion diffusion model project. This repository specifically handles the processing of Van Criekinge motion capture data, converting raw C3D files through a multi-stage pipeline into HumanML3D-compatible format suitable for training motion generation models.

### Related Repositories
- **LoRA-MDM (Primary)** - Main model repository
- **lora_mdm_age_dataset** - VC Pipeline v1 (deprecated)
- **lora_mdm_age_dataset_2** - VC Pipeline v2 (this repository)
- **LoRA-MDM-age** - Legacy code
- **motion_diffusion_model** - MDM original implementation

---

## Folder Structure

### Pipeline Outputs

| Folder | Description |
|:-------|:------------|
| `van_criekinge_unprocessed_1/` | Raw C3D dataset |
| `processed_markers_all_2/` | Output of Step 1 (preprocessed C3D data) |
| `fitted_smpl_all_3/` | Output of Step 2 (SMPL parameters) |
| `humanml3d_joints_4/` | Output of Step 3 (joint positions) |
| `Comp_v6_KLD01/` | Output of Step 4 (final HumanML3D format) |

### Supporting Folders

| Folder | Description |
|:-------|:------------|
| `smpl/` | SMPL body models |
| `utils/` | Utility functions |

---

## Key Files

### Pipeline Scripts

The pipeline consists of 4 sequential processing steps. **Note:** This v2 pipeline has the same functionality as v1, except `2_fit_smpl_markers.py` now produces **correct motion output** with fixed marker fitting.

| File | Purpose |
|:-----|:--------|
| `1_dataset_prep.py` | C3D preprocessing - converts raw C3D files to processed marker data |
| `2_fit_smpl_markers.py` | **Fixed** SMPL fitting - fits SMPL body model to marker data (see New Marker Fitting documentation) |
| `3_export_humanml3d.py` | Joint export - extracts HumanML3D joint positions from SMPL |
| `4_motion_process.py` | Feature extraction - converts joints to final 263-dimensional representation |

### Visualization & Debugging Tools

| File | Purpose |
|:-----|:--------|
| `6_npz_motion_to_gif.py` | Interactive 3D matplotlib viewer (22-joint HumanML3D skeleton) |
| `view_smpl_params.py` | SMPL parameter visualization (24-joint skeleton) |
| `render_smpl_mesh.py` | High-quality 3D mesh video renderer (full SMPL body with skinning) |
| `inspect_file.py` | View original C3D data as 3D plots |
| `explore_c3d.py` | C3D structure exploration |
| `debug.py` | SMPL shoulder joint verification |
| `check_dimensions.py` | Verify output has 263 dimensions |

---

## Pipeline Workflow

The processing pipeline follows these steps:

1. **Dataset Preparation** (`1_dataset_prep.py`)
   - Reads raw C3D motion capture files
   - Preprocesses marker data
   - Outputs to `processed_markers_all_2/`

2. **SMPL Fitting** (`2_fit_smpl_markers.py`)
   - Fits SMPL body model to preprocessed markers
   - **Fixed implementation** - corrects issues from v1
   - Outputs SMPL parameters to `fitted_smpl_all_3/`

3. **Joint Export** (`3_export_humanml3d.py`)
   - Extracts 22 HumanML3D joint positions from SMPL
   - Outputs to `humanml3d_joints_4/`

4. **Motion Processing** (`4_motion_process.py`)
   - Converts joints to 263-dimensional feature vectors
   - Final output to `Comp_v6_KLD01/`

---

## Key Improvements Over v1

- **Corrected SMPL marker fitting** - The primary fix in this v2 pipeline
- **Production-ready** - Validated output for training
- **Proper joint alignment** - Accurate motion representation

---

## Usage

To process the Van Criekinge dataset:

```bash
# Step 1: Preprocess C3D files
python 1_dataset_prep.py

# Step 2: Fit SMPL model (fixed version)
python 2_fit_smpl_markers.py

# Step 3: Export HumanML3D joints
python 3_export_humanml3d.py

# Step 4: Create final feature representation
python 4_motion_process.py
```

To visualize results:

```bash
# View final HumanML3D motion (22 joints)
python 6_npz_motion_to_gif.py

# View SMPL parameters (24 joints)
python view_smpl_params.py

# Render high-quality 3D mesh video
python3 render_smpl_mesh.py -s 01 -c 1  # Subject 01, Scene 1
python3 render_smpl_mesh.py -s 02 -c 0 --model female  # With specific model

# Inspect original C3D data
python inspect_file.py
```

---

## Requirements

- Python 3.x
- SMPL body models (in `smpl/` folder)
- Dependencies for C3D processing, SMPL fitting, and visualization
- See individual scripts for specific library requirements

---

## Documentation

For detailed information about the marker fitting corrections and technical implementation details, refer to the project wiki documentation on "New Marker Fitting".

---

## Dependencies & Installation

### Environment Setup (Recommended)

Use the provided `environment-v2.yml` to create a conda environment with ALL dependencies:

```bash
conda env create -f environment.yml
conda activate mdm-data-pipeline
```

This automatically installs:
- Python 3.10
- PyTorch 2.9.1 with CUDA 12.9
- All core scientific libraries (numpy, scipy, matplotlib)
- C3D file readers (c3d, ezc3d)
- SMPL body model library (smplx)
- 3D rendering tools (trimesh, pyrender, opencv)

### System Libraries (Optional but recommended for visualization)

For 3D rendering and visualization support:

```bash
# Basic rendering support
sudo apt-get install libopengl0 libglvnd0 libglx0 # Debian/Ubuntu
sudo dnf install libglvnd-opengl libglvnd-glx # RHEL/Fedora

# Advanced 3D mesh rendering
sudo apt-get install freeglut3-dev libglew-dev libglfw3-dev # Debian/Ubuntu
sudo dnf install freeglut-devel glew-devel glfw-devel # RHEL/Fedora
```

### SMPL Body Models

The SMPL body model files are required for `2_fit_smpl_markers.py` and `render_smpl_mesh.py`:

1. Download from https://smpl.is.tue.mpg.de/
2. Place in `smpl/` directory:
   - `SMPL_MALE.pkl`
   - `SMPL_FEMALE.pkl`
   - `SMPL_NEUTRAL.pkl`

---

## License

Part of the LoRA-MDM project for motion diffusion modeling research.
