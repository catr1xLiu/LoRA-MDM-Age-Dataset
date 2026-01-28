# AMASS: Archive of Motion Capture as Surface Shapes

**Authors:** Naureen Mahmood¹, Nima Ghorbani², Nikolaus F. Troje³, Gerard Pons-Moll⁴, Michael J. Black²

**Affiliations:**
- ¹ Meshcapade
- ² MPI for Intelligent Systems
- ³ York University
- ⁴ MPI for Informatics

**Contact:** nmahmood@meshcapade.com, {nghorbani, black}@tue.mpg.de, troje@yorku.ca, gpons@mpi-inf.mpg.de

**Website:** https://amass.is.tue.mpg.de/

## Overview

AMASS is a large-scale unified database of human motion that addresses the fragmentation in existing mocap datasets by:
1. Converting 15 different optical marker-based mocap datasets into a common SMPL representation
2. Developing MoSh++, an improved method to recover realistic 3D human meshes from sparse marker data
3. Capturing body shape, pose, soft-tissue dynamics, and hand motion

### Dataset Scale
- **Duration:** >40 hours of motion data
- **Subjects:** 344 individuals
- **Motions:** 11,265 distinct motion sequences
- **Source datasets unified:** 15 different mocap datasets
- **Marker sets:** 37 to 91 markers per dataset

## Motivation and Problem Statement

### Challenges in Existing Mocap Datasets
1. **Fragmentation:** Different markersets and laboratory-specific procedures
2. **Limited size:** Even largest single datasets are too small for serious deep learning
3. **Incompatible representations:** Each dataset uses different body parameterizations
4. **Loss of information:** Standard unification approaches normalize to fixed body proportions, altering the original data
5. **Incomplete data:** Most datasets lack 3D body shape information, containing only skeletons or markers

### Goals of AMASS
1. Create unified representation independent of original markerset
2. Maintain richness of original marker data
3. Include full 3D human mesh representation
4. Enable high-quality animation generation for synthetic training data
5. Support deep learning model development

## MoSh++ Method: Motion and Shape Capture

### Overview
MoSh++ is an extended and improved version of the original MoSh method that:
- Replaces SCAPE body model with SMPL for broader compatibility
- Captures realistic soft-tissue dynamics using DMPL
- Estimates hand pose using MANO model
- Fine-tunes hyperparameters using ground-truth 4D scan data

### Key Improvements Over MoSh

| Aspect | MoSh | MoSh++ |
|--------|------|--------|
| Body Model | BlendSCAPE (100 shape components) | SMPL (16 shape + 8 dynamics components) |
| Soft-tissue | Approximate (shape space changes) | Realistic (DMPL space) |
| Hand Modeling | None | SMPL-H with MANO model |
| Hyperparameter tuning | Limited | Optimized via SSM cross-validation |
| Graphics compatibility | Limited | Widely compatible (UV maps, blend skinning) |
| Accuracy (46 markers) | 12.1mm | 7.4mm |

## SMPL Body Model

### SMPL-H Model Structure
The SMPL-H model combines SMPL with hand articulation:

**Model Components:**
- **Template vertices (N):** 6,890 vertices in rest pose
- **Total joints:** 52 (22 body + 30 hand joints)
- **Shape parameters (β):** 16 dimensions
- **Pose parameters (θ):** 3 × 52 + 3 = 159 dimensions (includes 3D translation)
- **Dynamic shape parameters (φ):** 8 dimensions for soft-tissue motion

### SMPL Mathematical Formulation

```
S(β, θ, φ) = G(T(β, θ, φ), J(β), θ, W)

T(β, θ, φ) = T_μ + B_s(β) + B_p(θ) + B_d(φ)
```

Where:
- **G(T, J, θ, W):** Linear blend skinning function
- **T(β, θ, φ):** Template mesh in rest pose with deformations
- **T_μ:** Mean template
- **B_s(β):** Shape blend shapes (identity-dependent deformations)
- **B_p(θ):** Pose blend shapes (pose-dependent deformations)
- **B_d(φ):** Dynamic blend shapes (soft-tissue motion)
- **J(β):** Joint locations
- **W:** Blend skinning weights

### Model Variants

**SMPL-H:**
- Extended SMPL with hand model
- 52 joints total (22 body + 30 hand)
- Compatible with MANO hand model for detailed hand pose

**DMPL Integration:**
- Dynamic shape space from DYNA dataset
- Captures soft-tissue deformations in motion
- 8 dynamic coefficients
- Learned from 4D scans of subjects in motion

## MoSh++ Model Fitting

### Two-Stage Optimization Process

#### Stage I: Shape and Marker Location Estimation

**Objective Function:**
```
E(M̃, β, Θ_B, Θ_H) = λ_D E_D(M̃, β, Θ_B, Θ_H)
                      + λ_β E_β(β)
                      + λ_{θB} E_{θB}(θ_B)
                      + λ_{θH} E_{θH}(θ_H)
                      + λ_R E_R(M̃, β)
                      + λ_I E_I(M̃, β)
```

**Components:**
- **E_D:** Data term measuring distance between simulated and observed markers
- **E_β:** Mahalanobis distance shape prior on SMPL shape components
- **E_{θB}:** Body pose regularization
- **E_{θH}:** Hand pose regularization (24-D MANO pose space)
- **E_R:** Latent marker distance constraint (prescribed distance d = 9.5mm from body surface)
- **E_I:** Deviation penalty from initial marker locations

**Hyper-parameters (Final Weights):**
```
λ_D = 600 × b
λ_β = 1.25
λ_{θB} = 0.375
λ_{θH} = 0.125
λ_I = 37.5
λ_R = 1e4
```
Where `b = 46/n` (adjustment factor for marker set size; n = number of observed markers)

**Optimization Details:**
- **Frames used:** 12 randomly chosen frames per subject
- **Variables:** Poses Θ = {θ_1...F}, single shape β, latent marker positions M̃ = {m̃_i}
- **Soft-tissue:** Excluded from Stage I
- **Method:** Threshold Acceptance with 4 graduated optimization stages
- **Annealing strategy:** λ_D multiplied by s = 2 while dividing regularizers by same factor

#### Stage II: Per-Frame Pose and Dynamics Estimation

**Objective Function:**
```
E(θ_B, θ_H, φ) = λ_D E_D(θ_B, θ_H, φ)
                + λ_{θB} E_{θB}(θ_B)
                + λ_{θH} E_{θH}(θ_H)
                + λ_u E_u(θ_B, θ_H)
                + λ_φ E_φ(φ)
                + λ_v E_v(φ)
```

**Components:**
- **E_D:** Data term
- **E_{θB}:** Body pose prior
- **E_{θH}:** Hand pose prior
- **E_u:** Temporal smoothness for pose
- **E_φ:** Mahalanobis distance prior on DMPL coefficients
- **E_v:** Temporal smoothness for soft-tissue deformations

**Soft-tissue Dynamics Prior:**
```
E_φ(φ) = φ_t^T Σ_φ^{-1} φ_t
```
Where Σ_φ is diagonal covariance from DYNA dataset

**Hyper-parameters (Final Weights):**
```
λ_D = 400 × b
λ_{θB} = 1.6 × q
λ_{θH} = 1.0 × q
λ_u = 2.5
λ_φ = 1.0
λ_v = 6.0
```

**Missing Marker Adjustment Factor:**
```
q = 1 + ⌊x/|M| × 2.5⌋
```
Where:
- x = number of missing markers in frame
- |M| = total number of markers
- Range: 1.0 (all visible) to 3.5 (all missing)

**Optimization Process:**
1. **Initialization:** Rigid transformation between estimated and observed markers
2. **Graduated optimization:** Vary λ_{θB} from [10, 5, 1] times final weight
3. **Per-frame initialization:** Solution from previous frame
4. **Two-step per-frame optimization:**
   - First step: Remove dynamics terms, optimize pose only
   - Final step: Add dynamics and dynamics smoothness terms

### Hand Pose Estimation

**Hand Pose Space:**
- Uses MANO's 24-D low-dimensional PCA space (both hands combined)
- Projects full hand pose (90 parameters) into reduced space
- Applied during final two iterations of Stage I

**When Hand Markers Absent:**
- Hand poses set to average MANO pose

### Optimization Method

**Solver:** Powell's gradient-based dogleg minimization

**Implementation:** Chumpy auto-differentiation package

**Initialization Strategy:**
- Use Threshold Acceptance as fast annealing strategy
- Multiple graduated optimization stages
- Different strategies for first frame vs. subsequent frames

## SSM Dataset: Synchronized Scans and Markers

### Purpose
Created to:
1. Set hyperparameters for MoSh++ objective functions
2. Evaluate shape, pose, and soft-tissue motion reconstruction accuracy
3. Provide ground-truth 3D data with shape, pose, and dynamics variation

### Data Collection

**Mocap System:**
- 67 markers (optimized marker-set from original MoSh)
- OptiTrack mocap system

**4D Scanning:**
- 4D scanner synchronized with mocap
- Joint recording of surface geometry and marker positions

**Subject Pool:**
- 3 subjects with varying body shapes
- 30 different motion sequences
- 2 subjects professional models (signed modeling contracts enabling data release)

### Evaluation Protocol

**Ground Truth Measurement:**
- Uniform sampling: 10,000 points from 3D scans
- Metric: Scan-to-model distance (distance from each sampled point to closest surface on reconstructed mesh)
- Reported as: Mean of these distances (in mm)

### Testing Configurations

**Marker Set Evaluation:**
- Standard 46-marker subset of the 67-marker full set
- Full 67-marker set

**Data Splits:**
- Leave-one-subject-out cross-validation (4 iterations)
- Training: 2 subjects
- Testing: 1 subject (held-out)

## Hyper-Parameter Search Results

### Shape Estimation Accuracy
**Stage I Optimization:**
- Metric: Average error over 12 randomly selected frames
- Cross-validation: Leave-one-subject-out with 4 iterations
- Line search: On objective weights of Equation 3

**Results (46-marker set):**
- MoSh: 12.1mm
- MoSh++: 7.4mm
- Improvement: 39% reduction in error

**Accuracy Notes:**
- 16 shape components optimal (higher overfit to markers)
- 8 dynamic components optimal
- Achieved better accuracy than MoSh using 100 shape components

### Pose Estimation Accuracy
**Stage II Optimization:**
- Metric: Per-frame surface reconstruction error
- Data split: 20% held-out test set, 60% validation, 20% training
- Line search: On λ_D, λ_θ, λ_φ and missing-marker coefficient q

**Results Without Soft-tissue (46-marker set):**
- MoSh: 10.5mm
- MoSh++: 8.1mm
- Improvement: 23% error reduction

**Results With Soft-tissue (46-marker set):**
- MoSh: 10.24mm
- MoSh++: 7.3mm
- Improvement: 29% error reduction

**67-marker Set Performance:**
- MoSh++ with 46 markers nearly equals MoSh with 67 markers

## AMASS Dataset Composition

### Source Datasets Unified

| Dataset | Markers | Subjects | Motions | Duration (min) | Notes |
|---------|---------|----------|---------|----------------|-------|
| ACCAD | 82 | 20 | 252 | 26.74 | Ohio State University |
| BMLrub | 41 | 111 | 3,061 | 522.69 | Private collection |
| CMU | 41 | 96 | 1,983 | 543.49 | Carnegie Mellon University |
| EKUT | 46 | 4 | 349 | 30.74 | University Stuttgart |
| Eyes Japan | 37 | 12 | 750 | 363.64 | - |
| HumanEva | 39 | 3 | 28 | 8.48 | - |
| KIT | 50 | 55 | 4,232 | 661.84 | Karlsruhe Institute |
| MPI HDM05 | 41 | 4 | 215 | 144.54 | Max Planck Institute |
| MPI Limits | 53 | 3 | 35 | 20.82 | - |
| MPI MoSh | 87 | 19 | 77 | 16.53 | - |
| SFU | 53 | 7 | 44 | 15.23 | Simon Fraser University |
| SSM (New) | 86 | 3 | 30 | 1.87 | Synchronized Scans & Markers |
| TCD Hands | 91 | 1 | 62 | 8.05 | Trinity College Dublin |
| TotalCapture | 53 | 5 | 37 | 41.1 | - |
| Transitions | 53 | 1 | 110 | 15.1 | Custom recorded |

**Total AMASS:**
- **Combined duration:** 2,420.86 minutes (~40.3 hours)
- **Total subjects:** 344 individuals
- **Total motions:** 11,265 sequences

### Dataset Processing

**Quality Control:**
- Manual inspection of all results
- Correction or exclusion of problems (swapped/mislabeled markers)
- Verification of marker assignments

**Output Format:**
- All datasets converted to SMPL parameters
- Standardized representation across all source datasets
- Per-frame data includes:
  - SMPL 3D shape parameters (16 dimensions)
  - DMPL soft-tissue coefficients (8 dimensions)
  - Full SMPL pose parameters (159 dimensions, including hand articulation and global translation)

## Data Format and Usage

### Output Representation

**Per-Frame SMPL Parameters:**
1. **Shape parameters (β):** 16 dimensions (subject-specific, constant across motion)
2. **Pose parameters (θ):** 159 dimensions
   - Body pose: 22 joints × 3 = 66 dimensions
   - Hand pose: 30 joints × 3 = 90 dimensions
   - Global translation: 3 dimensions
3. **Dynamic shape parameters (φ):** 8 dimensions (per-frame soft-tissue)

**Flexibility Options:**
- Users caring only about pose: Ignore shape and soft-tissue components
- Users wanting normalized bodies: Use SMPL shape space to normalize to single shape
- Full richness: Use all components including soft-tissue

### Comparison to Traditional Mocap

**Traditional Datasets Include:**
- Skeletons (joints and kinematic chains)
- Or raw marker positions
- Or both

**AMASS Includes:**
- Fully rigged 3D meshes (6,890 vertices)
- SMPL skeleton with proper joints
- Shape information preserved
- Soft-tissue dynamics
- Hand articulation details

## Hand Articulation

### SMPL-H Hand Model Integration
- Extends SMPL with MANO hand model
- 30 hand joints (15 per hand)
- Fully compatible with SMPL for realistic animations
- Detailed hand pose when hand markers present

### Hand Motion Capture
**When Hand Markers Available:**
- MoSh++ solves for both body and hand pose
- Uses 24-D MANO pose space

**Quality and Realism:**
- More realistic hand poses than body-only estimation
- Richer animations with articulated hands

## Future Extensions

### Planned Improvements
1. **Extended SSM dataset:** Include captures with articulated hands
2. **Facial mocap:** Extend MoSh++ using SMPL-X model (face, body, hands together)
3. **Runtime optimization:** Parallel implementation using TensorFlow
4. **Automatic marker labeling:** Leverage AMASS for training denoising models
5. **Missing marker recovery:** Improve handling of occlusions

### Dataset Expansion
- Structured process for adding new captures
- New datasets convertible to SMPL format
- Community contribution framework in place

## Technical Specifications

### File Formats

**C3D Format:**
- Standard mocap file format
- Compatible with:
  - Vicon Nexus
  - Qualisys
  - Visual3D
  - MATLAB toolboxes (read_c3d)
  - Python libraries (ezc3d)

**SMPL Parameters:**
- Delivered as structured arrays or matrices
- Compatible with:
  - MATLAB
  - Python (with scipy.io)
  - Modern graphics engines

### Software Requirements

**Minimum Requirements:**
- MATLAB (or Python with scipy)
- Or any programming language with basic linear algebra libraries

**Optional Toolboxes:**
- MoSh code (for custom fitting)
- SMPL code (from SMPL website)
- Graphics software (Blender, Maya, Unity, Unreal Engine)

### Visualization and Rendering

**Supported Formats:**
- SMPL includes UV maps for texturing
- Compatible with:
  - Game engines (Unity, Unreal)
  - Graphics software (Blender, Maya)
  - Custom rendering pipelines

**Animation Pipeline:**
- Rigged SMPL mesh + pose parameters = full animation
- Real-time rendering in modern game engines
- Synthetic training data generation possible

## Related Work and Comparisons

### Previous Mocap Datasets
- Limited by small size (insufficient for deep learning)
- Varying body representations
- Different markersets and procedures
- Limited or no shape information

### Previous Unification Attempts
- Fixed body proportions (fundamentally alters data)
- Inverse kinematics normalization (changes motion)
- Loss of richness and realism

### MoSh++ Advantages
- Works with arbitrary markersets
- Preserves richness of original marker data
- Includes full 3D body shape
- Captures soft-tissue dynamics
- Includes hand articulation
- Enables realistic animation generation
- Suitable for synthetic training data creation

## Limitations

1. **Runtime:** MoSh++ not currently real-time (potential for optimization)
2. **Accuracy dependence:** Results vary with markerset quality and completeness
3. **Hand models:** Hand accuracy evaluation limited (no ground-truth hand data in SSM)
4. **Documentation:** Manual inspection required for marker assignments

## Key Contributions

1. **MoSh++ Method:** 
   - 39% improvement in shape estimation accuracy over MoSh
   - Realistic soft-tissue capture via DMPL
   - Hand pose estimation integration
   - Hyperparameter optimization via cross-validation

2. **AMASS Dataset:**
   - Unified representation of 15 datasets
   - 344 subjects, 11,265 motions, 40+ hours
   - Full 3D mesh representation with shape and soft-tissue
   - Rigged animation-ready models

3. **Practical Benefits:**
   - Large training dataset for deep learning
   - Compatible with standard graphics software
   - Animation and visualization ready
   - Maintains richness of original marker data
   - Synthetic training data generation enabled

## Impact and Applications

### Research Applications
- Deep learning for human motion synthesis
- Motion prediction and generation
- Gait analysis across datasets
- Body shape and motion analysis
- Soft-tissue dynamics modeling

### Practical Applications
- Character animation
- Virtual humans for graphics and games
- Motion retargeting
- Synthetic data generation for computer vision
- Biomechanical analysis

### Training Data for Deep Learning
- Sufficient size and diversity for neural networks
- Consistent representation across all samples
- Full 3D mesh information for training
- Ready for animation-based synthetic data generation

## Dataset Access and Citation

**Website:** https://amass.is.tue.mpg.de/

**Citation Format:**
```
Mahmood, N., Ghorbani, N., Troje, N. F., Pons-Moll, G., & Black, M. J. (2019). 
AMASS: Archive of Motion Capture as Surface Shapes. 
In International Conference on Computer Vision (ICCV).
```

**Community Support:**
- Research community access
- New dataset integration support
- Ongoing expansion and maintenance

## References

Key references for technical details:
- Allen, B., et al. (2003). The space of human body shapes
- Anguelov, D., et al. (2005). SCAPE: Shape Completion and Animation of People
- Black, M. J., & Jepson, A. D. (1998). EigenTracking robust matching of articulated objects
- Loper, M., et al. (2014). MoSh: Motion and Shape Capture from Sparse Markers
- Loper, M., et al. (2015). SMPL: A Skinned Multi-Person Linear Model
- Pons-Moll, G., et al. (2015). DYNA: A Model of Dynamic Human Shape in Motion
- Romero, J., et al. (2017). Embodied Hands: Modeling and Capturing Hands and Bodies Together (MANO)
