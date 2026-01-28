# LoRA-MDM Age Dataset Project

## Overview

This project processes the **Van Criekinge Gait Dataset** (188 healthy adults, ages 21-86) into HumanML3D format for machine learning. The goal is to train a motion diffusion model (LoRA-MDM) that can generate age-conditioned human motion from text descriptions.

## Current Status

**Week 3 Completed (Jan 2026):**
- Dockerized MoSh++ + HumanML3D pipeline established for reliable SMPL fitting
- Van Criekinge dataset cleaned and split into age groups (Young, Mid, Elderly)
- Replaced flawed Step 2→Step 3 conversion with validated pipeline
- Ready for motion clip re-processing and LoRA adapter training (Weeks 5-7)
- Target: Abstract submission Feb 15, full paper Weeks 8-9

**Key Achievement:** Shifted from continuous age conditioning to discrete age groups using LoRA adapters for realistic publishable results by mid-February.

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

## Folder Structure

### Data Directory

All data files are stored in the `data/` directory (gitignored). See [DATA.MD](DATA.MD) for download instructions.

| Folder | Description |
|:-------|:------------|
| `data/van_criekinge_unprocessed_1/` | Raw C3D dataset (download required) |
| `data/processed_markers_all_2/` | Output of Step 1 (preprocessed C3D data) |
| `data/fitted_smpl_all_3/` | Output of Step 2 (SMPL parameters) |
| `data/humanml3d_joints_4/` | Output of Step 3 (joint positions) |
| `data/Comp_v6_KLD01/` | Output of Step 4 (final HumanML3D format) |
| `data/smpl/` | SMPL body models (download required) |
| `data/outputs/` | Visualization outputs |

### Code Folders

| File | Purpose |
|:-----|:--------|
| `6_npz_motion_to_gif.py` | Interactive 3D matplotlib viewer (22-joint HumanML3D skeleton) |
| `view_smpl_params.py` | SMPL parameter visualization (24-joint skeleton) |
| `render_smpl_mesh.py` | High-quality 3D mesh video renderer (full SMPL body with skinning) |
| `inspect_file.py` | View original C3D data as 3D plots |
| `explore_c3d.py` | C3D structure exploration |
| `debug.py` | SMPL shoulder joint verification |
| `check_dimensions.py` | Verify output has 263 dimensions |

| Folder | Description |
|:-------|:------------|
| `utils/` | Utility functions |
| `visualize_joints/` | Joint visualization utilities |

### Documentation

See `docs/` directory for detailed information:
- **Van_Criekinge_Dataset.md**: Details of the Van Criekinge Dataset, including data structure and formats. Highly valuable.
- **AMASS_Archieve_Motion_Capture**: Details about the proccessing pipeline of the AMASS dataset, critical for understanding the software `Mosh++`, which fits smpl mesh onto MoCap markers.
- The `Legacy/` folder contains AI-generated note based on the previous studnet (Eugene)'s lab journal, containing:
   - **Legacy/Dataset Overview** : Additional Van Criekinge dataset information, mostly repeated
   - **Legacy/1 VC Pipeline/**: Processing pipeline stages and known fixes
   - **Legacy2 Closer Look at VC/**: Pipeline failure analysis and investigation
   - **Legacy/3 Solutions/**: Proposed solutions and marker fitting approaches
   - **Legacy/4 Codebase Adaptation**: Integration details and age conditioning architecture
   - **Legacy/Tools/**: Visualization and debugging utilities
   Most of them are not very useful, but `docs/Dataset Overview.md` and `docs/1 VC Pipeline/Pipeline Overview.md` are Must-Reads.

---

## Usage

### Setup

1. **Download required data** - See [DATA.MD](DATA.MD) for instructions on downloading:
   - Raw Van Criekinge C3D motion capture dataset
   - SMPL body models

2. **Process the Van Criekinge dataset:**

```bash
# Step 1: Preprocess C3D files
python 1_dataset_prep.py --data_dir data/van_criekinge_unprocessed_1 --output_dir data/processed_markers_all_2

# Step 2: Fit SMPL model (fixed version)
python 2_fit_smpl_markers.py --processed_dir data/processed_markers_all_2 --models_dir data/smpl --out_dir data/fitted_smpl_all_3

# Step 3: Export HumanML3D joints
python 3_export_humanml3d.py --fits_dir data/fitted_smpl_all_3 --out_dir data/humanml3d_joints_4

# Step 4: Create final feature representation
python 4_motion_process.py --build_vc --vc_root data --vc_splits_dir splits/
```

Run `python <script>.py --help` for detailed parameter documentation.

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

## License

Part of the LoRA-MDM project for motion diffusion modeling research.