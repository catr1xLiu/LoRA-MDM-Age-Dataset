# MoSh++ SMPL Conversion Attempt - Session Report

**Date:** January 31, 2026  
**Status:** Model conversion completed, blocked by SOMA container dependency issue  
**Next Action Required:** Fix SOMA container dependencies (see Section 3)

---

## 1. What Was Accomplished in This Session

### ✅ SMPL Model Conversion to MoSh++ Format

Successfully converted all SMPL models from NPZ to sparse PKL format compatible with MoSh++:

**Converted Models:**
- `data/smpl_mosh_fixed/smpl/neutral/model.pkl` (247MB)
- `data/smpl_mosh_fixed/smpl/male/model.pkl` (247MB)
- `data/smpl_mosh_fixed/smpl/female/model.pkl` (247MB)

**Key Conversions Applied:**
1. **J_regressor**: Converted from dense numpy array to scipy.sparse COO matrix
   - Shape: `(24, 6890)`
   - Non-zero elements: ~230
   - Now has `.row`, `.col`, `.data` attributes required by MoSh++

2. **Posedirs**: Preserved original shape `(6890, 3, 207)`
   - 207 parameters = 69 pose parameters (23 joints × 3) + 3 global orientation
   - Correct for SMPL model (MoSh++ expects 69 parameters)

3. **Added Required Fields:**
   - `bs_style = 'lbs'` (linear blend skinning)
   - `bs_type = 'lrotmin'` (rotation minimization)

4. **Pickle Protocol**: Saved with protocol 4 (Python 3.4+ compatible)

### ✅ Updated MoSh++ Integration Script

Modified `2_fit_smpl_markers_moshpp.py` to use SMPL instead of SMPL-X:

**Changes Made:**
- `surface_model.type`: `"smplx"` → `"smpl"`
- `opt_settings.weights_type`: `"smplx"` → `"smpl"`
- Output filename: `{trial}_smplx_params.npz` → `{trial}_smpl_params.npz`
- Metadata: `n_joints: 55` → `n_joints: 24`
- Metadata: `n_pose_params: 165` → `n_pose_params: 72`
- Documentation: Updated all references from SMPL-X to SMPL
- Default support directory: `/data/smplx` → `/data/smpl_mosh_fixed`

### ✅ Model Verification

All converted models verified to have:
- Sparse `J_regressor` with `.row` attribute ✓
- Correct `posedirs` shape for 69 SMPL parameters ✓
- Valid `kintree_table` with 24 joints ✓
- All required keys present ✓

---

## 2. Issues Encountered

### ❌ Critical Blocker: SOMA Container Dependency Issue

**Error:**
```python
Traceback (most recent call last):
  File "/project/2_fit_smpl_markers_moshpp.py", line 66, in run_mosh_on_c3d
    from soma.amass.mosh_manual import mosh_manual
  File "/root/soma/src/soma/amass/mosh_manual.py", line 33, in <module>
    from human_body_prior.tools.omni_tools import get_support_data_dir
ModuleNotFoundError: No module named 'human_body_prior.tools'
```

**Root Cause:**
The SOMA Docker container has `human-body-prior` version 2.2.2.0 installed, but this version is missing the `tools` submodule that `soma.amass.mosh_manual` requires.

**Evidence:**
```bash
$ podman run --rm localhost/soma:latest /bin/bash -lc ". /root/miniconda3/etc/profile.d/conda.sh && conda activate soma && pip list | grep -i human"
human-body-prior         2.2.2.0

$ podman run --rm localhost/soma:latest /bin/bash -lc ". /root/miniconda3/etc/profile.d/conda.sh && conda activate soma && python -c \"import human_body_prior; print(human_body_prior.__file__)\""
/root/miniconda3/envs/soma/lib/python3.7/site-packages/human_body_prior/__init__.py
```

**Impact:**
- Cannot run MoSh++ fitting until this dependency is resolved
- All model conversion work is complete but cannot be tested
- Blocking progress on the MoSh++ pipeline

### ⚠️ Minor Issues (Resolved)

1. **File Permissions**: Script had permission issues in container
   - **Solution**: Added execute permissions with `chmod +x`

2. **SELinux Context**: Files needed `:Z` flag for Podman
   - **Solution**: Used `-v "$(pwd):/project:Z"` syntax

3. **Python Command**: Container uses `python3` not `python`
   - **Solution**: Updated command to use `python3`

---

## 3. Task: Fix SOMA Container Dependencies

### Required Fix

Install the missing `human_body_prior.tools` module in the SOMA container.

### Test Code

Use this Python code to verify the dependency is correctly installed:

```python
#!/usr/bin/env python3
"""
Test script to verify human_body_prior.tools is properly installed in SOMA container.
Run this inside the SOMA container to check if dependencies are correctly set up.
"""

import sys

def test_human_body_prior_tools():
    """Test if human_body_prior.tools module is available."""
    print("=" * 60)
    print("Testing human_body_prior.tools availability")
    print("=" * 60)
    
    # Test 1: Import human_body_prior
    try:
        import human_body_prior
        print("✓ human_body_prior imported successfully")
        print(f"  Location: {human_body_prior.__file__}")
    except ImportError as e:
        print(f"✗ Failed to import human_body_prior: {e}")
        return False
    
    # Test 2: Import human_body_prior.tools
    try:
        from human_body_prior import tools
        print("✓ human_body_prior.tools imported successfully")
        print(f"  Location: {tools.__file__}")
    except ImportError as e:
        print(f"✗ Failed to import human_body_prior.tools: {e}")
        return False
    
    # Test 3: Import specific functions needed by MoSh++
    try:
        from human_body_prior.tools.omni_tools import get_support_data_dir
        print("✓ get_support_data_dir imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import get_support_data_dir: {e}")
        return False
    
    try:
        from human_body_prior.tools.omni_tools import makepath
        print("✓ makepath imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import makepath: {e}")
        return False
    
    try:
        from human_body_prior.tools.omni_tools import log2file
        print("✓ log2file imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import log2file: {e}")
        return False
    
    # Test 4: Import soma.amass.mosh_manual (the main module we need)
    try:
        from soma.amass.mosh_manual import mosh_manual
        print("✓ soma.amass.mosh_manual imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import mosh_manual: {e}")
        return False
    
    print("=" * 60)
    print("✓ ALL TESTS PASSED - Dependencies are correctly installed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_human_body_prior_tools()
    sys.exit(0 if success else 1)
```

### How to Run the Test

```bash
# Save the test script
cat > /tmp/test_deps.py << 'EOF'
[PASTE TEST CODE ABOVE HERE]
EOF

# Run in SOMA container
podman run --rm -v "/tmp/test_deps.py:/test_deps.py:Z" localhost/soma:latest /bin/bash -lc ". /root/miniconda3/etc/profile.d/conda.sh && conda activate soma && python3 /test_deps.py"
```

### Expected Output When Fixed

```
============================================================
Testing human_body_prior.tools availability
============================================================
✓ human_body_prior imported successfully
  Location: /root/miniconda3/envs/soma/lib/python3.7/site-packages/human_body_prior/__init__.py
✓ human_body_prior.tools imported successfully
  Location: /root/miniconda3/envs/soma/lib/python3.7/site-packages/human_body_prior/tools/__init__.py
✓ get_support_data_dir imported successfully
✓ makepath imported successfully
✓ log2file imported successfully
✓ soma.amass.mosh_manual imported successfully
============================================================
✓ ALL TESTS PASSED - Dependencies are correctly installed!
============================================================
```

### Suggested Fix Approaches

**Option A: Upgrade human-body-prior package**
```bash
podman run --rm -it localhost/soma:latest /bin/bash
# Inside container:
. /root/miniconda3/etc/profile.d/conda.sh && conda activate soma
pip install --upgrade human-body-prior
```

**Option B: Install from source with tools module**
```bash
podman run --rm -it localhost/soma:latest /bin/bash
# Inside container:
. /root/miniconda3/etc/profile.d/conda.sh && conda activate soma
pip uninstall human-body-prior -y
cd /root
# Clone and install from source if available
git clone https://github.com/nghorbani/human_body_prior.git
cd human_body_prior
pip install -e .
```

**Option C: Copy tools from local submodule**
```bash
# Mount the local human_body_prior submodule into the container
podman run --rm -it -v "$(pwd)/human_body_prior/human_body_prior/tools:/root/miniconda3/envs/soma/lib/python3.7/site-packages/human_body_prior/tools:Z" localhost/soma:latest /bin/bash
```

---

## 4. Detailed Plans for Next Steps

Once the dependency issue is fixed and the test passes, proceed with these steps:

### Step 1: Test MoSh++ with Single Subject (Subject 01, Trial 1)

**Command:**
```bash
podman run --rm \
    -v "$(pwd):/project:Z" \
    -v "$(pwd)/data:/data:Z" \
    localhost/soma:latest \
    /bin/bash -lc ". /root/miniconda3/etc/profile.d/conda.sh && conda activate soma && python3 /project/2_fit_smpl_markers_moshpp.py --c3d_path /data/van_criekinge_unprocessed_1/able_bodied/SUBJ01/SUBJ1_1.c3d --out_dir /data/fitted_smpl_all_3_moshpp --support_dir /data/smpl_mosh_fixed --gender neutral"
```

**Expected Output:**
- MoSh++ Stage I optimization completes successfully
- SMPL parameters saved to `/data/fitted_smpl_all_3_moshpp/SUBJ01/SUBJ1_1_smpl_params.npz`
- Betas saved to `/data/fitted_smpl_all_3_moshpp/SUBJ01/betas.npy`
- Metadata saved to `/data/fitted_smpl_all_3_moshpp/SUBJ01/SUBJ1_1_metadata.json`

**Success Criteria:**
1. No errors during MoSh++ execution
2. Output files are created and non-empty
3. `poses` array shape is `(n_frames, 72)` (69 body + 3 global)
4. `joints` array shape is `(n_frames, 24, 3)` (24 SMPL joints)

### Step 2: Validate Output Quality

**Verification Script:**
```python
import numpy as np

# Load MoSh++ output
mosh_data = np.load('data/fitted_smpl_all_3_moshpp/SUBJ01/SUBJ1_1_smpl_params.npz')

# Check shapes
print(f"Poses shape: {mosh_data['poses'].shape} (expected: (n_frames, 72))")
print(f"Trans shape: {mosh_data['trans'].shape} (expected: (n_frames, 3))")
print(f"Joints shape: {mosh_data['joints'].shape} (expected: (n_frames, 24, 3))")
print(f"Betas shape: {mosh_data['betas'].shape} (expected: (10,))")

# Check for NaN/Inf
print(f"\nNaN in poses: {np.isnan(mosh_data['poses']).any()}")
print(f"Inf in poses: {np.isinf(mosh_data['poses']).any()}")

# Compare with existing pipeline
existing_data = np.load('data/fitted_smpl_all_3/SUBJ01/SUBJ1_1_smpl_params.npz')
print(f"\nComparison with existing pipeline:")
print(f"Existing poses shape: {existing_data['poses'].shape}")
print(f"MoSh++ poses shape: {mosh_data['poses'].shape}")
```

### Step 3: Process Multiple Subjects

If single subject test passes, batch process all subjects:

```bash
# Create batch processing script
for subject_dir in data/van_criekinge_unprocessed_1/able_bodied/SUBJ*/; do
    subject_id=$(basename "$subject_dir")
    for c3d_file in "$subject_dir"/*.c3d; do
        echo "Processing: $c3d_file"
        podman run --rm \
            -v "$(pwd):/project:Z" \
            -v "$(pwd)/data:/data:Z" \
            localhost/soma:latest \
            /bin/bash -lc ". /root/miniconda3/etc/profile.d/conda.sh && conda activate soma && python3 /project/2_fit_smpl_markers_moshpp.py --c3d_path /data/van_criekinge_unprocessed_1/able_bodied/$subject_id/$(basename "$c3d_file") --out_dir /data/fitted_smpl_all_3_moshpp --support_dir /data/smpl_mosh_fixed --gender neutral --subject $subject_id"
    done
done
```

### Step 4: Compare MoSh++ vs Existing Pipeline

**Metrics to Compare:**
1. **Marker Reconstruction Error**: How well do fitted joints match input markers?
2. **Temporal Smoothness**: Are the motions smooth (no jitter)?
3. **Physical Plausibility**: Are poses anatomically valid?
4. **Processing Time**: How long does each take?

**Comparison Script:**
```python
import numpy as np
import matplotlib.pyplot as plt

def compare_pipelines(subject_id, trial_name):
    """Compare MoSh++ output with existing pipeline."""
    
    # Load both outputs
    mosh = np.load(f'data/fitted_smpl_all_3_moshpp/{subject_id}/{trial_name}_smpl_params.npz')
    existing = np.load(f'data/fitted_smpl_all_3/{subject_id}/{trial_name}_smpl_params.npz')
    
    # Compare pose differences
    pose_diff = np.abs(mosh['poses'] - existing['poses'])
    print(f"Mean pose difference: {pose_diff.mean():.4f} radians")
    print(f"Max pose difference: {pose_diff.max():.4f} radians")
    
    # Visualize differences
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(pose_diff.mean(axis=1))
    plt.title('Mean Pose Difference per Frame')
    plt.xlabel('Frame')
    plt.ylabel('Difference (radians)')
    
    plt.subplot(1, 2, 2)
    plt.hist(pose_diff.flatten(), bins=50)
    plt.title('Distribution of Pose Differences')
    plt.xlabel('Difference (radians)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'comparison_{subject_id}_{trial_name}.png')
    plt.close()

# Run comparison
compare_pipelines('SUBJ01', 'SUBJ1_1')
```

### Step 5: Integration with Downstream Pipeline

Once MoSh++ outputs are validated:

1. **Export to HumanML3D format** (Step 3):
   ```bash
   python 3_export_humanml3d.py --fits_dir data/fitted_smpl_all_3_moshpp --out_dir data/humanml3d_joints_4_moshpp
   ```

2. **Motion Processing** (Step 4):
   ```bash
   python 4_motion_process.py --build_vc --vc_root data --vc_splits_dir splits/
   ```

3. **Compare final outputs** with existing pipeline

### Step 6: Documentation and Cleanup

1. Update `AGENTS.md` with MoSh++ workflow
2. Document performance improvements (if any)
3. Archive old SMPL-X conversion attempts
4. Create troubleshooting guide for common MoSh++ issues

---

## Summary

**Status:** Ready to test once dependency issue is resolved

**Completed:**
- ✅ SMPL model conversion (all genders)
- ✅ MoSh++ script updates
- ✅ Model verification

**Blocked by:**
- ❌ SOMA container missing `human_body_prior.tools`

**Next Action:**
Fix SOMA container using test script in Section 3, then proceed with Step 1 above.

**Estimated Time to Complete:**
- Fix dependencies: 30 minutes
- Test single subject: 10 minutes
- Batch process all subjects: 4-6 hours (depending on dataset size)
- Validation and comparison: 2 hours
- **Total: ~1 day of work**
