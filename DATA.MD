# Data Directory Documentation

This document describes the data files required for this project and how to obtain them.

## Overview

All data files are stored in the `data/` directory, which is gitignored to keep the repository lightweight. The data includes raw motion capture recordings, SMPL body models, and intermediate processing outputs.

## Directory Structure

```
data/
├── van_criekinge_unprocessed_1/    # Raw C3D motion capture data (~1.3GB)
├── processed_markers_all_2/         # Processed marker data (~779MB) [generated]
├── fitted_smpl_all_3/               # SMPL fitted parameters (~21MB) [generated]
├── humanml3d_joints_4/              # HumanML3D format joints (~12MB) [generated]
├── Comp_v6_KLD01/                   # Motion features (~33MB) [generated]
├── smpl/                            # SMPL body models (~837MB)
└── outputs/                         # Visualization outputs [generated]
```

**[generated]** indicates directories created by running the pipeline scripts.

## Required Downloads

### 1. Raw Motion Capture Data (van_criekinge_unprocessed_1/)

**Source:** Van Criekinge et al. dataset published on Nature
**Download from:** [Figshare](https://figshare.com/)

**Dataset Details:**
- **Publication:** Van Criekinge, T., et al. "Lower limb muscle activation characteristics ..." *Nature Scientific Data* (2021)
- **Figshare DOI:** [10.6084/m9.figshare.c.5404560](https://doi.org/10.6084/m9.figshare.c.5404560)
- **Direct Link:** https://figshare.com/collections/Lower_limb_muscle_activation_characteristics/5404560

**Download Instructions:**
1. Visit the Figshare link above
2. Download the C3D motion capture files
3. Extract the archive to create the following structure:
   ```
   data/van_criekinge_unprocessed_1/
   ├── metadata.py          # Metadata about subjects
   ├── able_bodied/         # Able-bodied subject recordings
   │   ├── SUBJ01/
   │   │   ├── *.c3d       # Motion capture files
   │   ├── SUBJ02/
   │   └── ...
   └── stroke/              # Stroke patient recordings
       ├── TVC*/
       └── ...
   ```

**File Format:** C3D (Coordinate 3D) motion capture files
**Size:** ~1.3GB

### 2. SMPL Body Models (smpl/)

**Source:** SMPL - A Skinned Multi-Person Linear Model
**Download from:** [SMPL Official Website](https://smpl.is.tue.mpg.de/)

**License:** SMPL models are available for research purposes. You must register and agree to the license terms on the official website.

**Download Instructions:**
1. Visit https://smpl.is.tue.mpg.de/
2. Click "Download" and register for an account
3. Download the "SMPL for Python" package
4. Extract and copy the following files to `data/smpl/`:
   - `SMPL_NEUTRAL.npz` or `SMPL_NEUTRAL.pkl`
   - `SMPL_MALE.npz` or `SMPL_MALE.pkl`
   - `SMPL_FEMALE.npz` or `SMPL_FEMALE.pkl`
   - `J_regressor_extra.npy` (if available)
   - `kintree_table.pkl`
   - `smplfaces.npy`

**Required Structure:**
```
data/smpl/
├── SMPL_NEUTRAL.npz
├── SMPL_MALE.npz
├── SMPL_FEMALE.npz
├── SMPL_NEUTRAL.pkl      (optional, for compatibility)
├── SMPL_MALE.pkl         (optional, for compatibility)
├── SMPL_FEMALE.pkl       (optional, for compatibility)
├── J_regressor_extra.npy (optional)
├── kintree_table.pkl
└── smplfaces.npy
```

**Size:** ~837MB
**Note:** The project uses `.npz` format preferentially. If you download `.pkl` files, you may need to convert them.

## Data Pipeline

Once you have downloaded the required raw data, you can generate the processed data by running the pipeline scripts in order:

### Step 1: Process Raw C3D Data
```bash
python 1_dataset_prep.py \
    --data_dir data/van_criekinge_unprocessed_1 \
    --output_dir data/processed_markers_all_2
```
**Output:** `data/processed_markers_all_2/` - Cleaned and preprocessed marker trajectories

### Step 2: Fit SMPL Models
```bash
python 2_fit_smpl_markers.py \
    --processed_dir data/processed_markers_all_2 \
    --models_dir data/smpl \
    --out_dir data/fitted_smpl_all_3
```
**Output:** `data/fitted_smpl_all_3/` - SMPL body parameters (poses, shapes, translations)

### Step 3: Export to HumanML3D Format
```bash
python 3_export_humanml3d.py \
    --fits_dir data/fitted_smpl_all_3 \
    --out_dir data/humanml3d_joints_4
```
**Output:** `data/humanml3d_joints_4/` - 22-joint skeleton in HumanML3D format

### Step 4: Process Motion Features
```bash
python 4_motion_process.py \
    --build_vc \
    --vc_root data \
    --vc_splits_dir splits/ \
    --vc_meta_py data/van_criekinge_unprocessed_1/metadata.py
```
**Output:** `data/Comp_v6_KLD01/` - Motion features for training (263D feature vectors)

## Generated Data

These directories are created automatically when you run the pipeline:

- **processed_markers_all_2/**: Marker trajectories with gap filling and resampling
- **fitted_smpl_all_3/**: SMPL parameters fitted to marker data
  - Contains per-subject `betas.npy` (body shape)
  - Contains per-trial `*_smpl_params.npz` (poses and translations)
- **humanml3d_joints_4/**: 22-joint skeleton data at 20 FPS
- **Comp_v6_KLD01/**: Training-ready motion features
  - `train/`, `val/`, `test/` splits
  - `Mean.npy`, `Std.npy` normalization statistics
- **outputs/**: Visualization outputs from `render_smpl_mesh.py`

## Verifying Your Setup

After downloading the required data, verify the structure:

```bash
# Check if raw data exists
ls data/van_criekinge_unprocessed_1/able_bodied/SUBJ01/

# Check if SMPL models exist
ls data/smpl/*.npz

# Check directory structure
tree -L 2 data/
```

Expected output should show the `van_criekinge_unprocessed_1/` and `smpl/` directories with files inside.

## Troubleshooting

### Problem: Scripts can't find data directories

**Solution:** Make sure you're running scripts from the repository root and that the `data/` directory exists at the same level as the Python scripts.

### Problem: SMPL models not loading

**Solution:**
1. Verify that `.npz` files exist in `data/smpl/`
2. Check file permissions (files should be readable)
3. If you only have `.pkl` files, the code should still work but may require `chumpy`

### Problem: C3D files not found

**Solution:**
1. Verify the directory structure matches the expected layout
2. Ensure files are extracted, not still in `.zip` or `.tar.gz` format
3. Check that subject directories (SUBJ01, SUBJ02, etc.) contain `.c3d` files

### Problem: Import error for metadata.py

**Solution:**
1. Verify `data/van_criekinge_unprocessed_1/metadata.py` exists
2. Check that the file is a valid Python file
3. This file should come with the Van Criekinge dataset download

## Data Sizes

| Directory | Size | Type |
|-----------|------|------|
| van_criekinge_unprocessed_1 | ~1.3GB | Raw (required) |
| smpl | ~837MB | Raw (required) |
| processed_markers_all_2 | ~779MB | Generated |
| fitted_smpl_all_3 | ~21MB | Generated |
| humanml3d_joints_4 | ~12MB | Generated |
| Comp_v6_KLD01 | ~33MB | Generated |
| **Total** | **~3GB** | **~2.1GB raw + ~845MB generated** |

## SMPL Models for visualize_joints

The `visualize_joints/` directory (used for joints-to-SMPL mesh fitting) also requires SMPL models. After downloading SMPL models to `data/smpl/`, you have two options:

**Option 1: Create a symlink (recommended)**
```bash
# Remove the placeholder directory
rm -rf visualize_joints/smpl_models/smpl

# Create symlink to shared SMPL models
ln -s ../../data/smpl/smpl visualize_joints/smpl_models/smpl
```

**Option 2: Copy the models**
```bash
cp data/smpl/smpl/* visualize_joints/smpl_models/smpl/
```

The symlink approach saves ~837MB of disk space by sharing models between the main pipeline and visualization tools.

## Additional Notes

- All data directories are gitignored to keep the repository clean
- Generated data can be recreated by running the pipeline scripts
- Raw data (C3D files and SMPL models) must be downloaded separately due to licensing
- The pipeline is designed to be run incrementally - you can stop after any step
- Each script has `--help` documentation for detailed parameter information

## Citation

If you use the Van Criekinge dataset, please cite:

```bibtex
@article{vancriekinge2021,
  title={Lower limb muscle activation characteristics during walking in subjects with and without stroke},
  author={Van Criekinge, Tamaya and others},
  journal={Scientific Data},
  volume={8},
  number={1},
  pages={1--13},
  year={2021},
  publisher={Nature Publishing Group}
}
```

If you use SMPL models, please cite:

```bibtex
@article{loper2015smpl,
  title={{SMPL}: A skinned multi-person linear model},
  author={Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
  journal={ACM Transactions on Graphics (TOG)},
  volume={34},
  number={6},
  pages={1--16},
  year={2015},
  publisher={ACM}
}
```
