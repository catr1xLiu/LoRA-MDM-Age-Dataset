# LoRA-MDM Age Dataset Project

## Overview

This project processes the **Van Criekinge Gait Dataset** (188 healthy adults, ages 21-86) into HumanML3D format for machine learning. The goal is to train a motion diffusion model (LoRA-MDM) that can generate age-conditioned human motion from text descriptions.

## Dataset

- **Source**: Van Criekinge public gait dataset (kinematics, kinetics, EMG)
- **Participants**: 138 healthy adults + 50 stroke survivors
- **Format**: Raw C3D files and MATLAB (.mat) files
- **Coverage**: Lifespan data (21-86 years old)

## Processing Pipeline

The conversion follows 4 sequential stages:

1. **Data Preparation** (`tools/dataset_prep.py`): Clean C3D files, fill gaps, resample, filter markers
2. **SMPL Fitting** (`tools/fit_smpl_markers.py`): Fit 3D body model to marker positions
3. **HumanML3D Export** (`tools/export_humanml3d.py`): Canonicalize joints, resample to 20FPS
4. **Feature Processing** (`motion_process.py`): Convert to final `.npy` format for training

## Current Status

- ✅ Age conditioning integrated into MDM architecture
- ✅ `AgeMotionDataset` implemented for dataset loading
- ⚠️ Pipeline has known accuracy issues (see docs for details)

## Key Features

- **Age Normalization**: Ages mapped to [0, 1] range for model conditioning
- **Motion Canonicalization**: Root-centered, Y-up, Z-forward coordinates
- **LoRA Finetuning**: Efficient model adaptation using Low-Rank updates

## Documentation

See `docs/` directory for detailed information:
- **Dataset Overview**: Van Criekinge dataset structure and formats
- **1 VC Pipeline/**: Processing pipeline stages and known fixes
- **2 Closer Look at VC/**: Pipeline failure analysis and investigation
- **3 Solutions/**: Proposed solutions and marker fitting approaches
- **4 Codebase Adaptation**: Integration details and age conditioning architecture
- **Tools/**: Visualization and debugging utilities

Start with `docs/Dataset Overview.md` and `docs/1 VC Pipeline/Pipeline Overview.md` for context.
