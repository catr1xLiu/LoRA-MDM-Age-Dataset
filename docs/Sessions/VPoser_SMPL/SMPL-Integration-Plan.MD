# SMPL‑Integration Plan

**Location:** `docs/Sessions/SMPL-Integration-Plan.MD`  
**Purpose:**  Serve as a stand‑alone reference for anyone who needs to understand, maintain or extend the new SMPL fitting step that replaces `2_fit_smpl_markers.py`.

---

## 1. Overview  

The pipeline’s **Step 2 – SMPL fitting** originally used a custom implementation in `tools/fit_smpl_markers.py`.  
We have replaced it with a **drop‑in replacement** based on the **human_body_prior** library (specifically, the `IK_Engine` class).  The new script:

```
human_body_prior/src/fit_markers.py
```

- Reads cleaned marker files from `data/processed_markers_all_2/`.
- Uses the same CLI arguments as the legacy script.
- Maps all usable Plug‑in Gait markers to SMPL vertex IDs via `marker_vids['smpl']`.
- Performs batch fitting with **LBFGS** (300 iterations, 10 × betas) and a fixed set of stepwise weights.
- Saves output in the same `.npz` format expected by subsequent stages (`export_humanml3d.py`, `motion_process.py`).

The new implementation keeps *exactly* the same public API as the original, so no downstream changes are required.

---

## 2. Background Knowledge

| Topic | Key Points | Reference |
|-------|------------|-----------|
| **SMPL model** | 24 joints, 10 shape parameters (betas), Y‑up & Z‑forward coordinate system | `human_body_prior/body_model/body_model.py` |
| **Vicon → SMPL transformation** | Vicon uses Z‑up, Y‑forward. We rotate by –90° about X to obtain Y‑up, Z‑forward (`vicon_to_smpl_coords`). | `docs/Sessions/SMPL-Integration-Plan.MD#Coordinate‐System` |
| **Marker mapping** | Only markers with SMPL vertex IDs (33 Plug‑in Gait markers) are used.  Mapping defined in `human_body_prior/src/marker_vids.py`. | `human_body_prior/src/marker_vids.py` |
| **IK_Engine API** | `source_kpts` is a callable returning virtual markers from SMPL parameters. Optimizer loops over batches and applies stepwise weights. | `human_body_prior/models/ik_engine.py` |
| **Pipeline structure** | 1 → Process Raw C3D → 2 → Fit SMPL → 3 → Export HumanML3D → 4 → Feature extraction | `docs/Legacy/1 VC Pipeline/Pipeline Overview.md` |

---

## 3. Current Status

| Item | State | Notes |
|------|-------|-------|
| **Decision document** (`SMPL-Integration-Plan.MD`) | *Created* | Summarises decisions, plan & references |
| **New script skeleton** (`human_body_prior/src/fit_markers.py`) | *Pending implementation* | Will follow the architecture outlined below |
| **Marker mapping validation** | Completed | All 33 Plug‑in Gait markers map to SMPL vertices |
| **Coordinate transformation** | Re‑used from `vicon_to_smpl_coords` in existing code | Ensures consistency with legacy pipeline |
| **CLI compatibility** | Planned | Same arguments as `tools/fit_smpl_markers.py` |
| **Testing** | Pending | Will run against sample data and render with `render_smpl_mesh_live.py` |

---

## 4. Implementation Plan

### 4.1 File Layout

```
human_body_prior/
├── src
│   ├── fit_markers.py      # <‑ new drop‑in replacement
│   └── marker_vids.py
├── models
│   └── ik_engine.py
└── body_model
    └── body_model.py
```

### 4.2 Core Steps in `fit_markers.py`

1. **Argument parsing**  
   - Same options as the legacy script (`--processed_dir`, `--models_dir`, `--out_dir`, `--subject`, etc.).

2. **Subject handling**  
   - For each subject (or specified trial):
     * Load metadata JSON → determine gender.
     * Load SMPL model via `build_smpl()`.

3. **Marker preparation**  
   - Read NPZ (`load_markers_npz`).
   - Convert Vicon coordinates (`vicon_to_smpl_coords`).
   - Map marker names to canonical names using `labels_map.py`.
   - Keep only markers with vertex IDs (from `marker_vids['smpl']`).

4. **Batch fitting**  
   - Chunk sequences >128 frames → memory‑safe.
   - For each chunk:
     * Create `SourceKeyPoints` instance (`BodyModel`, vids, offset).
     * Build `IK_Engine` with fixed weights/optimizer (`LBFGS 300 iters`).
     * Run optimization per batch → obtain `betas`, `global_orient`, `body_pose`, `trans`.
   - Concatenate results across chunks.

5. **Output**  
   - Save in same `.npz` format: `poses`, `trans`, `betas`, `gender`, `subject_id`, `trial_name`, `fps`, `n_frames`, `joints`.

### 4.3 Optional Enhancements (Future Work)

| Feature | Status |
|---------|--------|
| Two‑pass fitting / outlier detection | Not required per user’s decision |
| Gaussian post‑smoothing | Skipped |
| Custom optimizer weights | Fixed defaults from example (`ik_example_mocap.py`) |

---

## 5. Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Marker mapping errors** | Fitting fails | Validate mapping table, log unmapped markers |
| **Memory blow‑up on long trials** | Crash | Chunking logic (max 128 frames) |
| **Coordinate mismatch** | Wrong pose orientation | Reuse `vicon_to_smpl_coords` from legacy code |
| **Output format drift** | Subsequent steps break | Strictly use same key names & shapes as legacy output |
| **Performance issues** | Slow processing | Leverage GPU (`cuda`) and batch size 128 |

---

## 6. References

- Pipeline overview: `docs/Legacy/1 VC Pipeline/Pipeline Overview.md`
- Marker mapping logic: `human_body_prior/src/marker_vids.py`
- Coordinate conversion: `docs/Sessions/SMPL-Integration-Plan.MD#Coordinate‑System` (snippet from `vicon_to_smpl_coords`)
- IK Engine implementation: `human_body_prior/models/ik_engine.py`
- Legacy script for comparison: `tools/fit_smpl_markers.py`

---

## 7. Next Steps

1. **Implement** `human_body_prior/src/fit_markers.py` following this plan.  
2. **Unit‑test** on a sample subject (e.g., `SUBJ01`, trial `0`).  
3. **Visual validation** using `render_smpl_mesh_live.py`.  
4. **Integrate** into the full pipeline (`data/fitted_smpl_all_3/`) and run downstream steps to confirm compatibility.

Once these steps are completed, the new SMPL fitting stage will be fully functional and replace the old implementation without breaking any part of the existing data processing workflow.