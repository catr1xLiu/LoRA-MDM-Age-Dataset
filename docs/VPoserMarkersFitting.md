# SMPL Fitting with VPoser IK_Engine

## Overview

The `fit_markers.py` script fits SMPL body model parameters to a single marker NPZ file using the human_body_prior library's IK_Engine and VPoser prior. For large-scale processing, `fit_batch.py` provides parallelized execution across subjects and trials.

## Installation and Setup

### 1. Activate UV Environment

```bash
cd human_body_prior
uv sync
```

### 2. Install pytorch3d (Special Instructions)

The `pytorch3d` library requires special installation due to compilation requirements:

```bash
# Limit to 4 parallel jobs (safe for 32GB RAM)
export MAX_JOBS=4 

# Install pytorch3d from source with no build isolation
uv pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation
```

**Note:** This installation may take 10-15 minutes depending on your system. The `--no-build-isolation` flag is required for proper compilation.

### 3. Download Required Models

Before running the scripts, ensure you have:

1. **VPoser v2.05 model** in `human_body_prior/support_data/dowloads/V02_05/`
   - Contains `.ckpt` files and `V02_05.yaml` configuration
   - Available from SMPL-X website

2. **SMPL body models** in `data/smpl/`
   - `SMPL_MALE.npz`, `SMPL_FEMALE.npz`, `SMPL_NEUTRAL.npz`
   - Available from SMPL website

## Usage

### Batch Processing (Recommended)

To process multiple subjects in parallel:

```bash
cd human_body_prior
uv run python -m src.fit_batch \
  --processed_dir ../data/processed_markers_all_2 \
  --models_dir ../data/smpl \
  --out_dir ../data/fitted_smpl_all_3 \
  --workers 8 \
  --subjects subj1-50
```

### Single File Processing

To process a specific marker file:

```bash
cd human_body_prior
uv run python -m src.fit_markers \
  --input ../data/processed_markers_all_2/SUBJ01/SUBJ1_0_markers_positions.npz \
  --output ../data/fitted_smpl_all_3/SUBJ01/SUBJ1_0_smpl_params.npz \
  --models_dir ../data/smpl
```

## Command Line Arguments

### `fit_batch.py`

| Argument | Description | Default |
|----------|-------------|---------|
| `--processed_dir` | Path to processed markers (output of Step 1) | Required |
| `--out_dir` | Root output directory for SMPL parameters | Required |
| `--models_dir` | Path to SMPL models directory | Required |
| `--subjects` | Subject range (e.g., `subj1-20`) | All subjects |
| `--workers` | Number of parallel processes | 8 |
| `--device` | Device to use ('cpu' or 'cuda') | 'cuda' |
| `--batch_size` | Maximum frames per batch | 128 |

### `fit_markers.py`

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to input marker NPZ | Required |
| `--output` | Path to output SMPL NPZ | Required |
| `--models_dir` | Path to SMPL models directory | Required |
| `--device` | Device to use ('cpu' or 'cuda') | 'cuda' |
| `--batch_size` | Maximum frames per batch | 128 |
| `--vicon_up` | Vicon up axis ('X', 'Y', or 'Z') | 'Z' |
| `--vicon_forward` | Vicon forward axis ('X', 'Y', or 'Z') | 'Y' |

## What Results Do We Get?

Each trial produces:

### 1. SMPL Parameters File (`*_smpl_params.npz`)

Contains all SMPL parameters:
- `poses`: (T, 72) - Root orientation + pose body
- `trans`: (T, 3) - Global translation
- `betas`: (10,) - Shape parameters
- `joints`: (T, 24, 3) - SMPL joint positions
- `gender`, `subject_id`, `trial_name`, `fps`, `n_frames`

### 2. Metadata File (`*_smpl_params.json`)

Contains fitting settings, markers used, and vertex ID mappings.

## How to Examine Results

### Using `render_smpl_mesh_live.py`

```bash
python render_smpl_mesh_live.py --subject 01 --scene 0 --model male
```

### Using `view_smpl_params.py`

```bash
python view_smpl_params.py --subject SUBJ01 --trial SUBJ1_0
```


## How the Script Works

### Program Execution Flow

```mermaid
graph TD
    A[Start: Parse CLI arguments] --> B[Setup device CPU/GPU]
    B --> C[Find subjects to process]
    C --> D{For each subject}
    D --> E[Load subject metadata]
    E --> F[Load SMPL model based on gender]
    F --> G{For each trial}
    G --> H[Load marker data from NPZ]
    H --> I[Map markers to SMPL vertex IDs]
    I --> J[Prepare markers for fitting]
    J --> K[Fit SMPL using IK_Engine]
    K --> L[Compute joints from SMPL]
    L --> M[Save results to NPZ and JSON]
    M --> N{More trials?}
    N --> G
    N --> O{More subjects?}
    O --> D
    O --> P[End: Processing complete]
```

### Key Functions

#### 1. `map_markers_to_vertex_ids(marker_names)`
- Maps Van Criekinge marker names to SMPL vertex IDs
- Uses `labels_map.py` for canonical name conversion
- Uses `marker_vids.py` for vertex ID lookup
- Only ~35/111 markers map to SMPL vertices (core Plug-in Gait markers)

#### 2. `vicon_to_smpl_coords(points, vicon_up, vicon_forward)`
- Converts coordinates from Vicon to SMPL system
- Default: Vicon (Z-up, Y-forward) → SMPL (Y-up, Z-forward)
- Applies appropriate rotation matrices

#### 3. `fit_sequence_with_ik_engine(markers_torch, vids, bm_fname, device, batch_size)`
- Core optimization using human_body_prior IK_Engine
- Uses VPoser v2.05 as pose prior
- L-BFGS optimizer with 300 max iterations
- Loss weights: data=10.0, poZ_body=0.01, betas=0.5
- Splits long sequences into batches for memory efficiency

#### 4. `create_source_keypoints(bm_fname, vids, device)`
- Creates PyTorch module that computes virtual markers from SMPL parameters
- Virtual markers = vertex positions + 0.0095m offset along normals
- Adapted from `ik_example_mocap.py`

### Marker Set Analysis & Mapping

This section details the marker configurations used in the Van Criekinge dataset and how they are mapped to the SMPL model.

#### 1. Raw Dataset Markers (Plug-In Gait)

The raw dataset utilizes a full-body marker set consisting of **111 markers**. This includes the standard 28 Plug-In Gait (PiG) markers plus additional tracking clusters (4 markers per segment) to ensure robustness during dynamic movements.

- **Head:** `LFHD`, `RFHD`, `LBHD`, `RBHD`
- **Torso:** `C7`, `T10`, `CLAV`, `STRN`
- **Pelvis:** `LASI`, `RASI`, `SACR`
- **Upper Limbs:** `LSHO`/`RSHO`, `LELB`/`RELB`, `LWRA`/`LWRB`, `RWRA`/`RWRB`, `LFIN`/`RFIN`
- **Lower Limbs:** `LTHI`/`RTHI`, `LKNE`/`RKNE`, `LTIB`/`RTIB`, `LANK`/`RANK`, `LHEE`/`RHEE`, `LTOE`/`RTOE`, `LMT5`/`RMT5`

#### 2. Comparison: Previous vs. Current Implementation

The pipeline has evolved from a direct joint-based fitting to a high-fidelity surface-based fitting.

| Feature | Previous (`2_fit_smpl_markers.py`) | Current (`fit_markers.py`) |
| :--- | :--- | :--- |
| **Method** | Direct Marker-to-Joint mapping | Marker-to-Vertex (IK_Engine) |
| **Marker Count** | 29 markers | ~35 markers |
| **Logic** | Estimated joint centers | Surface mesh vertex optimization |
| **Advantages** | Simple, fast | Anatomically correct, uses VPoser prior |
| **Limitations** | "Sliding" markers, inaccurate orientations | Computationally heavier (IK optimization) |

#### 3. Current Mapping Logic

The `fit_markers.py` script uses a two-step mapping to link raw markers to specific SMPL mesh vertex IDs:

1.  **Canonical Name Conversion** (via `labels_map.py`): Converts dataset-specific names to canonical AMASS/Mosh labels (e.g., `LASI` -> `LFWT`, `RASI` -> `RFWT`).
2.  **Vertex ID Lookup** (via `marker_vids.py`): Maps canonical labels to vertex indices on the SMPL mesh (e.g., `LFWT` -> Vertex `3021`).

#### 4. Markers Utilized in Fitting (~35 Markers)

The current implementation utilizes a strategic subset of markers to capture surface orientation and prevent joint "sliding":

- **Pelvis:** `LASI`/`RASI` mapped to Front Waist (`LFWT`/`RFWT`) vertices.
- **Wrists:** `LWRA`/`LWRB` mapped to Inner/Outer Wrist (`LIWR`/`LOWR`) vertices, providing better hand rotation.
- **Feet:** `LHEE`, `LTOE`, and `LMT5` are all mapped to their respective mesh locations, ensuring the foot model aligns with the ground plane.
- **Fingers:** `LFIN`/`RFIN` are included to constrain the hand pose.

**Surface Fitting Advantage:** By fitting to the mesh surface (with a 0.0095m marker offset), the `IK_Engine` produces much more anatomically correct results and utilizes the VPoser prior to prevent unnatural joint angles.

## Technical Details

### Coordinate Systems

- **Vicon Default**: Z-up, Y-forward
- **SMPL Default**: Y-up, Z-forward
- **Transformation**: Rotate -90° around X-axis

### Optimization Parameters

```python
# From ik_example_mocap.py
data_loss = torch.nn.MSELoss(reduction="sum")
stepwise_weights = [{"data": 10.0, "poZ_body": 0.01, "betas": 0.5}]
optimizer_args = {
    "type": "LBFGS",
    "max_iter": 300,
    "lr": 1,
    "tolerance_change": 1e-4,
    "history_size": 200,
}
```

### Output Format Compatibility

The script produces output identical to the original `2_fit_smpl_markers.py`:

| Feature | Original | New Script |
|---------|----------|------------|
| Poses shape | (T, 72) | (T, 72) |
| Joints shape | (T, 24, 3) | (T, 24, 3) |
| Betas shape | (10,) | (10,) |
| Metadata | Included | Included + enhanced |
| Coordinate system | SMPL (Y-up, Z-forward) | SMPL (Y-up, Z-forward) |

## Troubleshooting

### Common Issues

1. **"No module named 'pytorch3d'"**
   - Install pytorch3d as described in Installation section
   - Use `--no-build-isolation` flag

2. **"No checkpoint found at ..."**
   - Ensure VPoser v2.05 model is downloaded and extracted
   - Check that `.ckpt` files exist in `V02_05/snapshots/`

3. **"SMPL model file not found"**
   - Download SMPL models from SMPL website
   - Place in `data/smpl/` directory

4. **Memory issues with long sequences**
   - Reduce `--batch_size` (default: 128)
   - Use `--device cpu` if GPU memory is limited

5. **Visualization not working**
   - Check EGL/OpenGL dependencies are installed
   - For headless servers, ensure libEGL is available

### Performance Tips

- **GPU vs CPU**: Use `--device cuda` for 5-10x speedup (if available)
- **Batch size**: Adjust based on sequence length and memory
- **Downsampling**: Consider downsampling marker data if frames > 1000
- **Parallel processing**: Process different subjects in parallel

## Integration with Pipeline

This script replaces Step 2 in the Van Criekinge processing pipeline:

1. **Step 1**: `1_dataset_prep.py` → Raw C3D to processed markers
2. **Step 2**: `fit_markers.py` → Markers to SMPL parameters (replaces `2_fit_smpl_markers.py`)
3. **Step 3**: `3_export_humanml3d.py` → SMPL to HumanML3D joints
4. **Step 4**: `4_motion_process.py` → Joints to 263D feature vectors

To continue the pipeline with new fits:

```bash
# After running fit_markers.py
python 3_export_humanml3d.py \
  --fits_dir data/fitted_smpl_all_3_new \
  --out_dir data/humanml3d_joints_4_new

python 4_motion_process.py --build_vc --vc_root data --vc_splits_dir splits/
```

## References

1. **human_body_prior library**: https://github.com/nghorbani/human_body_prior
2. **VPoser paper**: "Expressive Body Capture: 3D Hands, Face, and Body from a Single Image"
3. **SMPL model**: https://smpl.is.tue.mpg.de/
4. **Van Criekinge dataset**: Healthy adults gait dataset (ages 21-86)
5. **Original pipeline**: Eugene's LoRA-MDM project documentation