>[!info|right]
> **Part of a series on**
> ###### [[Dataset Overview|Van Criekinge]]
> ---
> **Pipeline**
> * [[1 VC Pipeline/Pipeline Overview|Pipeline Overview]]
> * [[1 VC Pipeline/dataset_prep.py fixes|Dataset Prep]]
> * [[1 VC Pipeline/motion_process.py fixes|Motion Process]]

>[!info]
> ###### VC Pipeline
> [Type::Data Pipeline]
> [Input::C3D Format]
> [Output::HumanML3D Format]
> [Status::Under Revision]

The **Van Criekinge Pipeline** is a multi-stage data processing workflow designed to convert raw C3D motion capture data from the Van Criekinge dataset into a tailored format compatible with the HumanML3D training framework. It involves data cleaning, SMPL parameter fitting, and feature extraction.

## Process Stages
The pipeline consists of four sequential scripts, each transforming the data representation.

### 1. Data Preparation
**Script**: `tools/dataset_prep.py`
*   **Input**: Raw C3D files.
*   **Function**: Performs gap filling, resampling, and filtering of non-position markers.
*   **Output**: Cleaned marker positions (`*_markers_positions.npz`).


### 2. SMPL Fitting
**Script**: `tools/fit_smpl_markers.py`
*   **Input**: Cleaned marker positions.
*   **Function**:
    *   Selects a subset of anatomical markers.
    *   Initializes SMPL parameters.
    *   Runs optimization loop (`fit_sequence`) to adjust SMPL parameters (pose, shape, translation) to minimize distance between model joints and input markers.
*   **Output**: SMPL parameters and derived joint positions (`*_smpl_params.npz`).

### 3. HumanML3D Export
**Script**: `tools/export_humanml3d.py`
*   **Input**: SMPL joint arrays.
*   **Function**: Selects the first 22 joints, canonicalizes them (root-centered, Y-up, Z-forward), and resamples to 20FPS.
*   **Output**: Canonicalized joint data (`*_humanml3d_22joints.npz`).

### 4. Feature Processing
**Script**: `data_loaders/humanml/scripts/motion_process.py`
*   **Command**: Run with `--build_vc`.
*   **Function**: Converts the canonicalized data into the final `.npy` feature format required for training.

---

## Known Issues
> [!warning] Pipeline Errors
> This pipeline has demonstrated inaccuracies in the final output, see [[1 Current Pipeline Failures|Pipeline Failures]]
> ![[correct_walk_npz.gif|center]]


