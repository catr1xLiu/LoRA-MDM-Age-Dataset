# SMPL→NTU-25 Import: T-Pose Exclusion, Frame Trimming & Full Dataset Validation

<aside>
📋

This page documents the fixes made to `import_from_smpl.py` that exclude T-pose calibration trials, trim instrumentally noisy boundary frames, and add robust filename parsing — raising walking detection accuracy from 75% to 100% across the full 138-subject Van Criekinge dataset.

</aside>

---

## What Changed

| Area | Before | After |
| --- | --- | --- |
| Trial 0 (T-pose) | Included — static standing pose with no gait motion | Skipped — detected by trial index extracted from filename |
| Boundary frames | Raw full-length clips fed to SMPL | First and last 5 frames trimmed before forward pass |
| Filename parsing | `stem.split('_')[1]` — crashes on non-standard names | Compiled regex `SUBJ\d+_(\d+)_smpl_params` — skips unrecognised files with a warning |
| Dataset size | 13 subjects (incomplete upload), 56 clips | 138 subjects (complete), 450 clips |

---

## Problems & Fixes

### Trial 0: T-Pose Contamination

Every subject's first capture (`_0_smpl_params.npz`) is a static T-pose used for SMPL shape fitting calibration — not a walking trial. Including it in the dataset injects clips where the subject is standing still with arms outstretched, producing skeleton sequences that look nothing like gait. The ST-GCN++ model correctly rejects them, predicting actions like `shake head` (label 35) instead of any walking label.

**Fix:** `get_trial_number()` extracts the trial index from the filename using a regex. Any file with index `0` is counted as discarded and skipped before loading.

### Boundary Frame Trimming

The Van Criekinge instrumentation introduces inaccurate SMPL fits at the start and end of each trial (~5 frames each) due to sensor initialisation and shutdown transients. These frames produce anatomically implausible joint positions that add noise to the skeleton sequences.

**Fix:** `smpl_to_ntu25()` slices `poses` and `trans` arrays to `[5:-5]` before running the SMPL forward pass. ST-GCN++ wraps shorter sequences back to 100 frames automatically, so no padding logic is needed.

### Malformed Filename: SUBJ101

The complete dataset contains one file without a trial index in its name (`SUBJ101_smpl_params.npz`). The original `split('_')[1]` parser returned `"smpl"` for this file, raising a `ValueError` and crashing the entire import.

**Fix:** The regex `^SUBJ\d+_(\d+)_smpl_params$` only matches files with an explicit trial number. Non-matching files print a `[SKIP]` warning and are counted in `n_discarded`.

---

## Inference Results — Before vs After

| Condition | Val Accuracy | Train Accuracy | Subjects | Clips |
| --- | --- | --- | --- | --- |
| Before fixes (incomplete dataset) | 75.0% (6/8) | — | 13 | 56 |
| After fixes (incomplete dataset) | 100.0% (6/6) | 100.0% (37/37) | 13 | 43 |
| After fixes (complete dataset) | **100.0% (87/87)** | **100.0% (363/363)** | 138 | 450 |

All top-1 predictions on the complete dataset fell into NTU walking labels: 58 (`walking towards`), 59 (`walking apart`), and 115 (`follow`).

---

## Dataset Statistics (Complete)

| Age Group | Clips |
| --- | --- |
| Young (< 40) | 155 |
| Adult (40–64) | 172 |
| Elderly (≥ 65) | 123 |
| **Total** | **450** |

138 T-pose trials and 1 malformed file were discarded (139 total).

---

## Operational Notes

- Inference on the full dataset exceeds the login node CPU time limit. Submit via SLURM to `TrixieMain` under account **`jpn-302`**.
- The `rsync` one-liner for syncing the dataset from WSL to Trixie: `rsync -avz --progress /mnt/c/Users/liuyir/Documents/LoRA-MDM-Age-Dataset/data/fitted_smpl_all_3_tuned/ liuyir@hn1.trixie.hpc:/gpfs/projects/AIP/jpn-302/LoRA-MDM-Age-Dataset/data/fitted_smpl_all_3_tuned/`

---

## Files Changed

- **`stgcnpp/import_from_smpl.py`** — added `FRAME_TRIM`, `_TRIAL_RE`, `get_trial_number()`; updated `collect_clips()` to return `(clips, n_discarded)`; updated `smpl_to_ntu25()` to trim boundary frames; added `import re`
