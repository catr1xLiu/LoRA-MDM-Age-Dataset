# A Full-Body Motion Capture Gait Dataset of 138 Able-Bodied Adults Across the Life Span and 50 Stroke Survivors

**Citation:** Tamaya Van Criekinge, Wim Saeys, Steven Truijen, Luc Vereeck, Lizeth H. Sloot, Ann Hallemans (2023)

**Journal:** Scientific Data, Volume 10, Article 852

**DOI:** https://doi.org/10.1038/s41597-023-02767-y

## Overview

This dataset contains comprehensive biomechanical data of 138 able-bodied adults (ages 21-86) and 50 stroke survivors walking at their preferred speed. The dataset is unique for its size, population diversity, and multimodal measurements.

### Dataset Scope
- **Able-bodied participants:** 138 individuals (ages 21-86 years)
- **Stroke survivors:** 50 individuals (ages 19-85 years)
- **Motion type:** Bare-footed walking at preferred speed
- **Measurements:** Full-body kinematics, kinetics, and muscle EMG activity

### Data Measurements
- **Kinematics:** Full-body 3D marker trajectories, joint angles, center of mass
- **Kinetics:** Ground reaction forces, joint forces, joint moments, joint power
- **Electromyography (EMG):** 14 back and lower limb muscles (bilaterally)
- **Additional:** Anthropometric measurements, clinical descriptors for stroke survivors

## Dataset Characteristics

### Able-Bodied Adults (n=138)

| Parameter | Mean (SD) | Range |
|-----------|-----------|-------|
| Age (years) | 51 (20) | 21-86 |
| Gender | 65M/73F | - |
| Body mass (kg) | 74 (15) | 48-157 |
| Height (mm) | 1684 (103) | 1420-1920 |
| Body Mass Index | 26 (4) | 18-47 |
| Leg length (mm) | 899 (61) | 660-1070 |

### Stroke Survivors (n=50)

| Parameter | Mean (SD) | Range |
|-----------|-----------|-------|
| Age (years) | 64 (14) | 19-85 |
| Gender | 34M/16F | - |
| Body mass (kg) | 72 (14) | 50-106 |
| Height (mm) | 1705 (80) | 1500-1890 |
| Body Mass Index | 25 (4) | 18-35 |
| Leg length (mm) | 882 (46) | 790-970 |
| Time post-stroke (days) | 53 (19) | 14-139 |
| Stroke type | 39 Ischemic / 11 Hemorrhagic | - |
| Lesion location | 17 Left / 33 Right | - |
| Functional Ambulation Category | 3 (1) | 2-5 |
| Trunk Impairment Scale | 14 (3) | 7-20 |
| Tinetti POMA | 19 (6) | 6-28 |

## Participant Inclusion/Exclusion Criteria

### Able-Bodied Adults - Exclusion Criteria
- Self-reported visual impairments
- Antalgic gait pattern
- Abnormal mobility in lower limbs
- Known neurological or orthopaedic disorder affecting motor performance and balance
- No requirement for medical records

### Stroke Survivors - Inclusion Criteria
1. Confirmed haemorrhagic or ischaemic stroke diagnosis (via CT or MRI imaging)
2. No known history of previous stroke
3. Stroke onset within five months
4. Age between 18 and 85 years

### Stroke Survivors - Exclusion Criteria
1. Trunk Impairment Scale ≥20 (indicating normal truncal function)
2. Functional Ambulation Score >2 (requirement: able to ambulate without continuous physical support)
3. Unable to sit independently for 30 seconds without support
4. Other neurological or orthopaedic disorders affecting motor performance
5. Unable to understand instructions

**Ethical Approval:** Declaration of Helsinki compliance, Ethics Review Committee reference numbers: 15/42/433, 151203ACADE, B300201316328

**Trial Registration:** ClinicalTrials.gov ID: NCT02708888

## Instrumentation and Data Collection

### Laboratory Setup
**Location:** Multidisciplinary Motor Centre Antwerp (M2OCEAN)

### Motion Capture System
- **System:** 8 Vicon T10 passive marker cameras
- **Sampling rate:** 100 frames per second
- **Resolution:** 1 Megapixel (1120 × 896)
- **Markers:** 14mm reflective markers (B&L Engineering, California)
- **Model:** Plug-In Gait full body model (standard 28 bony anatomical landmarks)

### Force Plate System
- **Type:** Ground-embedded force plates
- **Sampling rate:** 1000 frames per second
- **Plates:** 
  - 3x AMTI type OR 6-7 (46 × 50 × 8 cm)
  - 1x AccuGait® (50 × 50 × 4 cm)
- **Force threshold:** 10N for event detection

### Electromyography (EMG)
- **System:** 16-channel telemetric wireless surface EMG (Zerowire, Cometa)
- **Electrodes:** Polymer Ag/AgCl coated circular gel electrodes (Covidien Kendall™, 30mm × 24mm)
- **Configuration:** Bipolar sensor arrangement
- **Electrode placement:** Following SENIAM guidelines
- **Inter-electrode distance:** 20mm
- **Muscle coverage (bilateral):**
  - Rectus femoris (RF)
  - Vastus lateralis (VL)
  - Biceps femoris (BF)
  - Semitendinosus (ST)
  - Tibialis anterior (TA)
  - Gastrocnemius (GAS)
  - Erector spinae (ERS)

### Plug-In Gait Model
The Plug-In Gait model uses 28 markers placed on standardized bony anatomical landmarks:
- Includes markers for pelvis, lower limbs, torso, arms, and head
- Based on Davis et al. (1991) methodology
- Provides full-body 3D joint kinematics

## Data Collection Protocol

### Pre-Participant Preparation
- Anthropometric measurements: body mass, height, leg length
- Leg length measured as longitudinal distance between spina iliaca anterior superior and malleoli medialis (supine position)

### For Stroke Survivors (Additional Measurements)
- Time since stroke
- Stroke type
- Lesion location
- Functional Ambulation Categories (FAC)
- Trunk Impairments Scale (TIS)
- Tinetti Performance Oriented Mobility Assessment
- Measurements on different day to prevent fatigue

### EMG Electrode Placement
1. 14 electrode placement locations palpated and confirmed by selective muscle contractions
2. For stroke survivors unable to perform selective contraction: placement based solely on SENIAM guidelines
3. Skin preparation: shaving and degreasing to ensure good electrode-skin contact
4. Double-sided tape and skin-sensitive tape for secure fixation
5. Signal-to-noise ratio checked in Vicon software by maximal muscle contraction

### Marker Placement and Calibration
1. Reflective markers attached to bony anatomical landmarks (Plug-In Gait model)
2. Manual palpation by trained investigator
3. Double-sided tape and skin-sensitive tape for firm fixation
4. Static calibration performed with knee alignment device
5. Base pose: standing still in center of force platform with arms extended to sides, thumbs down

### Walking Protocol
- **Walkway:** 12-meter pathway
- **Speed:** Self-selected preferred speed
- **Equipment:** Walking without aids or orthoses
- **Instructions:** "Walk as usual, not to mind the markers, look forward, don't interact with anyone"
- **Safety:** For stroke survivors, skilled physiotherapist walked beside participant (non-hemiplegic side) providing no assistance
- **Marker visibility:** Maintained throughout trials

### Data Requirements
- **Target:** Minimum 6 walking trials during steady-state walking phase
- **Heel strikes:** 3 correct strikes on force plates from right and left foot (without other foot contact)
- **Exclusion of:** Gait initiation and termination phases
- **Adaptation for stroke/elderly:** Some participants had reduced trial numbers due to fatigue (2 subjects with only 2 dynamic C3D files)
- **Assistance:** 3 stroke survivors required minimal physical assistance (hand-holding during walking)

## Data Processing

### Marker Trajectory Processing
1. **Tracking and labeling:** Vicon Nexus software (versions 1.8.5 to 2.10.1)
2. **Gap filling:** 
   - Manual pattern or spline fills
   - Maximum limit: 20 consecutive frames
   - No automated pipelines used
3. **Quality control:** Manual visual checking of all trajectories

### Gait Event Detection
1. **Foot strike detection:** Based on ankle and heel marker trajectories
2. **Toe-off detection:** Based on ankle marker trajectory
3. **Force plate validation:** Force threshold of 10N
4. **Manual verification:** All gait events visually checked before processing

### Kinematics Calculation
1. **Filter:** 4th-order reversed Butterworth filter
2. **Cutoff frequency:** 10Hz low-pass filter
3. **Pipeline:** Plug-In Gait Dynamic pipeline in Vicon Nexus
4. **Smoothing:** Woltring filter based on 5th-order interpolating function with MSE smoothing value of 10
5. **Output:** 
   - 3D full-body joint angles
   - 3D center of mass
   - 3D joint forces, moments, and power

### EMG Processing
1. **Rectification:** EMG data rectified
2. **Filtering:**
   - 2nd-order reversed Butterworth bandpass filter
   - Passing frequencies: 10-300Hz
3. **Envelope creation:** 50ms moving average filter
4. **Normalization:** To maximum value found across available strides per muscle
5. **Time normalization:** 1000 datapoints per stride

### Kinetic Data Processing
1. **Filter:** 4th-order reversed Butterworth filter with 10Hz low-pass frequency
2. **Normalization:** Time-normalized to 1000 datapoints per stride
3. **Quality selection:** 
   - Able-bodied: Only strides with good force plate landings (manually verified)
   - Stroke: All available kinematic strides processed

### Final Data Output
1. **Walking direction:** Determined from heel data
2. **Corrections:** Anterior-posterior and medio-lateral direction corrections applied
3. **MAT structure creation:** Gait events and relevant data added to MATLAB structure

## Data Records and File Formats

### Available Data Files

#### Source Data (C3D Format)
- **Folders:** 
  - `50_StrokePiG` - stroke survivor data
  - `138_HealthyPiG` - able-bodied adult data
- **File naming:**
  - Able-bodied: `SUBJ(Number)` for dynamic trials, `SUBJ(0)` for static trial
  - Stroke survivors: `BWA_No` for dynamic trials, `[cal No]` for static trials
- **Content per file:**
  - Anthropometric measurements
  - 3D marker trajectories
  - 3D joint angles
  - 3D center of mass
  - Ground reaction forces
  - 3D joint forces, moments
  - 1D joint power
  - Unfiltered EMG

#### Post-Processed Data (MAT Format)
Two versions available for able-bodied and stroke survivors:

**Version 1 - Full stride-normalized data:**
- Filename: `MAT_normalizedData_AbleBodiedAdults_v06-03-23` and `MAT_normalizedData_PostStrokeAduls_v27-02-23`
- Contains:
  - Stride-normalized kinematic data (3D joint angles, center of mass, marker positions)
  - EMG data (raw and normalized traces)
  - Kinetic data (normalized to body mass: ground reaction forces, joint moments, joint powers)
  - Gait events
  - Anthropometric data
  - All available strides

**Version 2 - Summary data:**
- Filename: `MAT_normalizedData_AbleBodiedAdults_v06-03-23` and `MAT_normalizedData_PostStrokeAdults_v27-02-23`
- Contains:
  - Time-normalized average per participant per variable

#### Supporting Files
- **Description file:** `MATdatafles_description_v1.3_LST` - Detailed explanation of MAT-file structure
- **Stride count file:** `NrStrides` - Number of strides with good kinematic and kinetic data
- **Visual presentations:** PNG figures of selected variables with corresponding MATLAB code

#### Data Quality Notes
- **Able-bodied participants:** 
  - Average 6 good strides for kinematic data
  - Average 2 good strides for kinetic data (left and right combined)
- **Stroke participants:**
  - Average 8 good strides for kinematic data (paretic and non-paretic side)
  - Average 1.5 good strides for kinetic data

### Data Completeness

#### EMG Data Availability
- **Able-bodied with complete EMG:** 111/138 participants
- **Stroke survivors with complete EMG:** 47/50 participants
- **Missing EMG:** 
  - Able-bodied: 27 subjects (SUBJ23, 48, 54, 72, 85, and SUBJ118-138)
  - Stroke: 3 subjects (TVC11, TVC55, TVC57)

#### Assistance During Walking
- **Stroke survivors requiring minimal physical support:** 3 subjects (TVC48, TVC51, TVC54)
- **Support method:** Hand-holding of non-hemiplegic hand by physiotherapist

## Technical Validation

### System Accuracy and Precision

#### Vicon T10 System Performance
**Calibration Protocol:**
- Dummy with 5 markers (14mm diameter) in 90° pre-defined configuration
- Testing at 4 locations: beginning, middle (2 orientations), and end of walkway
- Marker orientations tested: sagittal, frontal, transversal, and 3D dynamic
- 5 recordings of 2.00 seconds duration each per condition

**Accuracy Results:**
- **Angular measurement accuracy (static):** 0.95mm (SD 0.85mm) and 0.138° (SD 0.07°)
- **Angular measurement accuracy (dynamic):** 1.81mm (SD 2.88mm) and 0.12° (SD 0.04°)
- **System error (baseline):** <2.0mm

**Precision Results:**
- **Distance precision (static):** 0.05mm (SD 0.02mm)
- **Distance precision (dynamic):** 0.54mm (SD 0.18mm)
- **Angular precision (static):** 0.02° (SD 0.003°)
- **Angular precision (dynamic):** 0.27° (SD 0.20°)

### Plug-In Gait Model Reliability
- Good intra-protocol repeatability (well-established in literature)
- Commonly used for full-body gait analysis
- Main variance contributor: Differences in marker placement

### Marker Placement Quality Control
- **Single investigator responsibility:** Primary researcher (TVC) performed all marker placement
- **Training:** Extensive training in anatomical landmark palpation
- **Fixation method:** Double-sided tape and skin-sensitive tape for consistent placement
- **Replacement protocol:** If marker fell off, immediate re-attachment at same position using marker print as guide
- **Recalibration:** Only one instance required full static recalibration (marker location undeterminable)

### Data Processing Quality Control
1. **Visual inspection:** Data checked at every processing step
2. **No automation in labeling/gap-filling:** Manual visual checking used throughout
3. **Single researcher processing:** One experienced researcher (TVC) performed post-processing in Vicon Nexus
4. **Validated software:** Used validated and reliable software for variable calculations
5. **Gait event verification:** All events visually checked for accuracy
6. **Trajectory reconstruction:** Full 3D trajectories reconstructed without gaps before further processing

### Kinetic Data Quality
- **Able-bodied participants:** Clean strikes checked and verified during collection
- **Stroke participants:** Clean strikes not guaranteed; user should verify
- **Able-bodied MAT-files:** Kinetic data checked for quality and outliers with manual corrections
- **Stroke MAT-files:** Kinetic analysis not a priority; outliers not assessed

### EMG Data Quality Control
**SENIAM Guidelines Compliance:**
- Standard sensor equipment (Ag/AgCl coated electrodes)
- Skin preparation: Dry, degreased (diethylether), and shaved
- Inter-electrode distance: 20mm
- Direction: Aligned with muscle fiber direction
- Placement location:** Longitudinal direction, halfway between motor endplate and tendon
- Researcher consistency:** Same experienced researcher (TVC) placed all electrodes
- Muscle verification:** Participants asked to contract muscle to verify electrode placement location
- Cable management:** No stretch on wireless cables; firmly attached with double-sided tape
- Signal-to-noise ratio:** Checked in Vicon software by asking participants to maximally contract each muscle (if possible)

## Usage Notes

### File Access and Import

#### C3D Files
- Format: Standard C3D file format (https://www.c3d.org)
- Compatibility: 
  - Motion capture software: Vicon Nexus, Qualisys, Visual3D
  - Programming languages: MATLAB, Python, C++ with "read C3D toolbox"
- Supports combination of different data types and lengths

#### MAT Files
- Primary software: MATLAB
- Alternative access: Python with additional steps and toolboxes
- Available tools: MatToPy tool on GitHub for importing MATLAB data to Python
  - GitHub: https://github.com/LizSloot/Readin_MatlabStruct_toPython.git

### Code Availability
- Example MATLAB code provided: `ExampleCode_LoadStruct_PlotTimeNormVar_v1`
- Freely available toolboxes:
  - Dumas toolbox (3D kinematics and inverse dynamics)
  - Too toolbox
  - Advance Gait Workflow from Vicon website
  - EMG Feature Extraction Toolbox

### Data Repository
All data available at: **Figshare**
- DOI: https://doi.org/10.6084/m9.figshare.c.6503791.v1

## Applications and Future Perspectives

### Current Applications
Dataset has been instrumental in showing:
- Age-related decline in peak muscle amplitude
- Decline in center of mass push-off power with age
- Age-related reductions in joint range of motion
- Mediolateral margins of stability throughout adulthood
- Trunk impairments during walking after stroke

### Research Questions Addressable
- Aging processes during walking
- Typical and hemiplegic gait comparison
- Age-matched controls for normal walking
- Speed-independent gait analysis (note: only self-selected speeds included)

### Potential Applications
- Gait and balance training optimization for stroke survivors
- Aging adult movement optimization
- Assistive robotics for elderly care (age-specific movement understanding)
- Deep learning model training for motion generation
- Clinical gait analysis reference data

## Limitations

1. **Speed analysis not possible:** Only self-selected speeds included
2. **Kinetic data in stroke dataset:** Not prioritized; quality not guaranteed
3. **EMG availability:** Missing for 27 able-bodied and 3 stroke survivors
4. **Marker loss:** 2 participants (TVC42, TVC56) with only 2 dynamic C3D files

## Key References

- Davis, R. B., et al. (1991). A gait analysis data collection and reduction technique
- Grood, E. S., & Suntay, W. J. (1983). A joint coordinate system for clinical description of 3D motions
- Vicon documentation and technical specifications (versions 1.8.5 to 2.10.1)
- SENIAM guidelines for surface electromyography
- MathWorks MATLAB R2021a
