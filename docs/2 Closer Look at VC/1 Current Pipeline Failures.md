>[!info|right]
> **Part of a series on**
> ###### [[Dataset Overview|Van Criekinge]]
> ---
> **Analysis**
> * [[2 Closer Look at VC/1 Current Pipeline Failures|Pipeline Failures]]
> * [[2 Closer Look at VC/2 Explore VC C3D|C3D Exploration]]
> * [[2 Closer Look at VC/3 Explore VC MATLAB|MATLAB Exploration]]

>[!info]
> ###### Pipeline Failures
> [Type::Issue Log]
> [Severity::High]
> [Status::Investigating]

This document outlines the **critical failures** observed in the initial Van Criekinge processing pipeline. Visual inspection of the processed motion data revealed significant anatomical and trajectory errors.
 

## Observed Issues
The following video demonstrates the pipeline failure:
![[pipeline_failure_proof.mov]]

**Key Anomalies**:
1.  **Sideways Pelvis**: The pelvis orientation is incorrect.
2.  **Jiggly Spine**: The spine exhibits unnatural, high-frequency noise.
3.  **Forward Lean**: The character leans forward excessively.
4.  **Descending Y-Axis**: The character appears to sink downwards over time.

### Resolution Status
*   **Issues 1 & 2 (Pelvis/Spine)**: Addressed in [[New Marker Fitting]].
*   **Issues 3 & 4 (Lean/Descent)**: Analyzed below.


---

## Root Cause Analysis (Issues 3 & 4)
Further investigation revealed that the forward lean and descent are **expected behaviors** of the canonicalization process, not pipeline errors.

**Cause**: `3_export_humanml3d.py` rotates every sequence into a **canonical body frame**:
*   **Up**: +Y (aligned from pelvis to neck)
*   **Forward**: +Z (cross product of up and hips)
*   **Left**: +X (from hips)

**Mechanism**:
1.  Estimates body axes from joints based on anatomy (not the walkway).
2.  Builds a fixed rotation matrix `R` for the entire sequence.
3.  Applies `R` to both joints and the *global pelvis path*.
4.  Root-centers the joints (pelvis at origin) while saving the global path separately.

This alignment causes the appearance of "sinking" or "leaning" if the original motion was on a slope or had a specific biological orientation that differed from the calculated canonical frame.