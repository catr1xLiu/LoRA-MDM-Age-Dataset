>[!info|right]
> **Part of a series on**
> ###### [[Dataset Overview|Van Criekinge]]
> ---
> **Pipeline**
> * [[1 VC Pipeline/Pipeline Overview|Pipeline Overview]]
> * [[1 VC Pipeline/dataset_prep.py fixes|Dataset Prep]]
> * [[1 VC Pipeline/motion_process.py fixes|Motion Process]]

>[!info]
> ###### Dataset Prep Fix
> [Type::Code Maintenence]
> [Script::dataset_prep.py]
> [Issue::Marker Naming Inconsistency]
> [Status::Resolved]

This document details a **code fix** implemented in `dataset_prep.py` to handle inconsistent marker naming conventions found in the Van Criekinge C3D files, which previously caused pipeline crashes.


## Problem Description
It was discovered that certain C3D files (e.g., trials for `SUBJ03`) contained marker names with inconsistent prefixes, such as `19301106v:LFHD` instead of the standard `LFHD`. This caused two major issues:
1.  **Pipeline Crash**: `fit_smpl_markers.py` failed with `RuntimeError: Too few anatomical markers found` because it could not match the labels.
2.  **Metadata Corruption**: The `key_markers` field in the JSON metadata was empty, as the logic relied on exact string matches.

## Implemented Solution
The solution involved modifying `dataset_prep.py` to "sanitize" marker names at the point of ingestion by stripping prefixes.


### 1. Cleaning Snippet
The following logic was added to the `create_marker_layout_config` method to strip whitespace, convert to uppercase, and remove any prefix separated by a colon.

```python
names_original = sample_trial_data['marker_names']
cleaned_names = []
for m in names_original:
    m_clean = m.strip().upper()
    if ":" in m_clean:
        m_clean = m_clean.split(":")[-1]
    cleaned_names.append(m_clean)
```

### 2. Implementation Details

#### Propagating Cleaned Names
The `cleaned_names` list replaced the original names list throughout the layout configuration:
- Updated `total_markers` and `marker_names` fields.
- Refactored loops for identifying `body_parts` and `key_markers` to use `cleaned_names`.

```python
layout['key_markers'] = {
    'head': [i for i, n in enumerate(cleaned_names) if n in ['LFHD', 'RFHD', 'LBHD', 'RBHD']],
    'spine': [i for i, n in enumerate(cleaned_names) if n in ['C7', 'T10', 'SACR']],
    'pelvis': [i for i, n in enumerate(cleaned_names) if n in ['LASI', 'RASI', 'SACR']],
    # ...
```

#### Metadata Synchronization
To ensure the cleaned names were preserved in the final output, a line was added to sync the metadata object with the layout's cleaned names before saving:

```python
meta['marker_names'] = layout['marker_names']
```

## Results
After the patch, all processed marker files exhibited consistent naming conventions. The `key_markers` fields were correctly populated, and `2_fit_smpl_markers.py` was able to process all trials (including `SUBJ3_1`) without runtime errors.

