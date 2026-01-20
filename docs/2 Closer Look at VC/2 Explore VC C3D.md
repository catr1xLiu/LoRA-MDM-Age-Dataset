>[!info|right]
> **Part of a series on**
> ###### [[Dataset Overview|Van Criekinge]]
> ---
> **Analysis**
> * [[2 Closer Look at VC/1 Current Pipeline Failures|Pipeline Failures]]
> * [[2 Closer Look at VC/2 Explore VC C3D|C3D Exploration]]
> * [[2 Closer Look at VC/3 Explore VC MATLAB|MATLAB Exploration]]

>[!info]
> ###### C3D Exploration
> [Type::Data Investigation]
> [Tool::Python / ezc3d]
> [Format::C3D]

This document details the **exploration of the raw C3D file structure** from the Van Criekinge dataset. The goal is to understand the stored parameters, identified points, and data organization using Python.


## File Structure Analysis
Using `ezc3d`, we inspected a sample file: `./138_HealthyPiG_10.05/SUBJ01/SUBJ1 (1).c3d`.

```python
from ezc3d import c3d
c = c3d("./138_HealthyPiG_10.05/SUBJ01/SUBJ1 (1).c3d")
print(c["parameters"]["POINTS"])
```

**Output (Points List)**:
The file contains a mix of standard anatomical markers (e.g., `LFHD`, `C7`) and calculated biomechanical outputs (Angles, Powers, Forces, Moments).

```python fold:"Full Point List"
['LFHD', 'RFHD', 'LBHD', 'RBHD', 'C7', 'T10', 'CLAV', 'STRN', 'LSHO', 'LELB', 'LWRA', 'LWRB', 'LFIN', 'RSHO', 'RELB', 'RWRA', 'RWRB', 'RFIN', 'LASI', 'RASI', 'SACR', 'LTHI', 'LKNE', 'LTIB', 'LANK', 'LHEE', 'LTOE', 'RTHI', 'RKNE', 'RTIB', 'RANK', 'RHEE', 'RTOE', 'CentreOfMass', 'CentreOfMassFloor', 'HEDO', 'HEDA', 'HEDL', 'HEDP', 'LCLO', 'LCLA', 'LCLL', 'LCLP', 'LFEO', 'LFEA', 'LFEL', 'LFEP', 'LFOO', 'LFOA', 'LFOL', 'LFOP', 'LHNO', 'LHNA', 'LHNL', 'LHNP', 'LHUO', 'LHUA', 'LHUL', 'LHUP', 'LRAO', 'LRAA', 'LRAL', 'LRAP', 'LTIO', 'LTIA', 'LTIL', 'LTIP', 'LTOO', 'LTOA', 'LTOL', 'LTOP', 'PELO', 'PELA', 'PELL', 'PELP', 'RCLO', 'RCLA', 'RCLL', 'RCLP', 'RFEO', 'RFEA', 'RFEL', 'RFEP', 'RFOO', 'RFOA', 'RFOL', 'RFOP', 'RHNO', 'RHNA', 'RHNL', 'RHNP', 'RHUO', 'RHUA', 'RHUL', 'RHUP', 'RRAO', 'RRAA', 'RRAL', 'RRAP', 'RTIO', 'RTIA', 'RTIL', 'RTIP', 'RTOO', 'RTOA', 'RTOL', 'RTOP', 'TRXO', 'TRXA', 'TRXL', 'TRXP', 'LHipAngles', 'LKneeAngles', 'LAnkleAngles', 'LAbsAnkleAngle', 'RHipAngles', 'RKneeAngles', 'RAnkleAngles', 'RAbsAnkleAngle', 'LPelvisAngles', 'RPelvisAngles', 'LFootProgressAngles', 'RFootProgressAngles', 'LShoulderAngles', 'LElbowAngles', 'LWristAngles', 'RShoulderAngles', 'RElbowAngles', 'RWristAngles', 'LNeckAngles', 'RNeckAngles', 'LSpineAngles', 'RSpineAngles', 'LHeadAngles', 'RHeadAngles', 'LThoraxAngles', 'RThoraxAngles', 'LAnklePower', 'RAnklePower', 'LKneePower', 'RKneePower', 'LHipPower', 'RHipPower', 'LWaistPower', 'RWaistPower', 'LNeckPower', 'RNeckPower', 'LShoulderPower', 'RShoulderPower', 'LElbowPower', 'RElbowPower', 'LWristPower', 'RWristPower', 'LGroundReactionForce', 'RGroundReactionForce', 'LNormalisedGRF', 'RNormalisedGRF', 'LAnkleForce', 'RAnkleForce', 'LKneeForce', 'RKneeForce', 'LHipForce', 'RHipForce', 'LWaistForce', 'RWaistForce', 'LNeckForce', 'RNeckForce', 'LShoulderForce', 'RShoulderForce', 'LElbowForce', 'RElbowForce', 'LWristForce', 'RWristForce', 'LGroundReactionMoment', 'RGroundReactionMoment', 'LAnkleMoment', 'RAnkleMoment', 'LKneeMoment', 'RKneeMoment', 'LHipMoment', 'RHipMoment', 'LWaistMoment', 'RWaistMoment', 'LNeckMoment', 'RNeckMoment', 'LShoulderMoment', 'RShoulderMoment', 'LElbowMoment', 'RElbowMoment', 'LWristMoment', 'RWristMoment']
```


## Data Extraction
We accessed `c['data']['points']`, a numpy array with shape ($3 \times N \times F \times 1$):
*   $3$: X, Y, Z coordinates.
*   $N$: Total number of points.
*   $F$: Total number of frames.

### Example: Left Hip Angles
Extracting Index 111 (`LHipAngles`) reveals the angular data structure:

```python
all_points = c['data']['points']
LHipAngles_data = all_points[:, 111, :]
print(LHipAngles_data)
```

**Output**:
```
[[ 20.65888405  20.3088131   19.85797119 ...   7.83167315   7.59125996
    7.53613043] # <- X Axis Values
 [  6.90810823   7.17020702   7.4846487  ...   9.02871799   9.04385281
    9.04731274] # <- Y Axis Values
 [-12.76604939 -12.70482254 -12.58753872 ...  -7.56184244  -7.47851562
   -7.4601965 ] # <- Z Axis Values
 [  1.           1.           1.         ...   1.           1.
    1.        ]] # <- Confidence
```

Where:
*   **X axis**: Flexion (Extension) (Swinging leg forwards and backwards)
*   **Y axis**: Adduction (swinging leg out to the side)
*   **Z axis**: Internal/External Rotation (twisting the thigh)

### Plug-in Gait (PiG) Definitions
According to the [Vicon Plug-in Gait Reference](https://help.vicon.com/download/attachments/11378719/Plug-in%20Gait%20Reference%20Guide.pdf), we validated specific markers:
*   **LTIO** (Index 63): Left Tibia Origin (Knee Joint Centre).
*   **LFEO**: Left Femur Origin.


At index 63, printing gives

```
[[-9.29452271e+02 -9.29493286e+02 -9.29553955e+02 ...  2.57873779e+03
   2.57878345e+03  2.57879468e+03]
 [ 2.57722076e+02  2.57605347e+02  2.57463440e+02 ...  2.28433884e+02
   2.28427643e+02  2.28426300e+02]
 [ 7.00033493e+01  6.99186096e+01  6.98224716e+01 ...  5.45469475e+01
   5.45507355e+01  5.45517159e+01]
 [ 1.00000000e+00  1.00000000e+00  1.00000000e+00 ...  1.00000000e+00
   1.00000000e+00  1.00000000e+00]]
```

Similarly, LFEO is Left Femur Origin 



---

## Skeleton Reconstruction
We defined a connection map to verify the skeleton structure.

```python
skeleton = [
    ("PELO", "TRXO"), ("TRXO", "HEDO"), # Spine
    ("TRXO", "LCLO"), ("TRXO", "RCLO"), # Shoulders
    ("LCLO", "LHUO"), ("LHUO", "LRAO"), ("LRAO", "LHNO"), # Left Arm
    ("RCLO", "RHUO"), ("RHUO", "RRAO"), ("RRAO", "RHNO"), # Right Arm
    ("PELO", "LFEO"), ("LFEO", "LTIO"), ("LTIO", "LFOO"), ("LFOO", "LTOO"), # Left Leg
    ("PELO", "RFEO"), ("RFEO", "RTIO"), ("RTIO", "RFOO"), ("RFOO", "RTOO"), # Right Leg
]
```

### Joints Hierarchy
```text
# Spine
Pelvis Origin <-> Thorax Origin
Thorax Origin <-> Head Origin

# Shoulders
Thorax Origin <-> Left Clavicle Origin
Thorax Origin <-> Right Clavicle Origin

# Left Arm
Left Clavicle Origin <-> Left Humerus Origin
Left Humerus Origin <-> Left Radius Origin
Left Radius Origin <-> Left Hand Origin

# Right Arm
Right Clavicle Origin <-> Right Humerus Origin
Right Humerus Origin <-> Right Radius Origin
Right Radius Origin <-> Right Hand Origin

# Left Leg
Pelvis Origin <-> Left Femur Origin
Left Femur Origin <-> Left Tibia Origin
Left Tibia Origin <-> Left Foot Origin
Left Foot Origin <-> Left Toe Origin

# Right Leg
Pelvis Origin <-> Right Femur Origin
Right Femur Origin <-> Right Tibia Origin
Right Tibia Origin <-> Right Foot Origin
Right Foot Origin <-> Right Toe Origin
```


---

## Visualization Script (`inspect_file.py`)
A comprehensive script was written to animate the skeleton with a follow-cam.

```python
from ezc3d import c3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

c = c3d("./138_HealthyPiG_10.05/SUBJ01/SUBJ1 (1).c3d")

# for i, label in enumerate(c['parameters']['POINT']["LABELS"]["value"]):
#     print(f"Index: {i}, Label: {label}")

all_points = c['data']['points']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Get all labels
labels = c['parameters']['POINT']["LABELS"]["value"]

# Helper to get point data by label
def get_point_data(label):
    if label in labels:
        idx = labels.index(label)
        return all_points[:, idx, :] # All frames
    return None

# Define skeleton connections
skeleton = [
    ("PELO", "TRXO"), ("TRXO", "HEDO"), # Spine
    ("TRXO", "LCLO"), ("TRXO", "RCLO"), # Shoulders
    ("LCLO", "LHUO"), ("LHUO", "LRAO"), ("LRAO", "LHNO"), # Left Arm
    ("RCLO", "RHUO"), ("RHUO", "RRAO"), ("RRAO", "RHNO"), # Right Arm
    ("PELO", "LFEO"), ("LFEO", "LTIO"), ("LTIO", "LFOO"), ("LFOO", "LTOO"), # Left Leg
    ("PELO", "RFEO"), ("RFEO", "RTIO"), ("RTIO", "RFOO"), ("RFOO", "RTOO"), # Right Leg
]

# Pre-fetch data for all connections
lines_data = []
for p1_name, p2_name in skeleton:
    p1_data = get_point_data(p1_name)
    p2_data = get_point_data(p2_name)
    if p1_data is not None and p2_data is not None:
        lines_data.append((p1_data, p2_data))

# Initialize lines
lines = [ax.plot([], [], [], marker='o', color='blue')[0] for _ in lines_data]

# Calculate global limits for consistent scaling
all_x = []
all_y = []
all_z = []
for p1_data, p2_data in lines_data:
    all_x.extend(p1_data[0, :])
    all_x.extend(p2_data[0, :])
    all_y.extend(p1_data[1, :])
    all_y.extend(p2_data[1, :])
    all_z.extend(p1_data[2, :])
    all_z.extend(p2_data[2, :])

# Calculate a fixed bounding box size for the camera
# We want a box big enough to fit the skeleton at any frame, but centered on the skeleton.
# Let's find the max extent of the skeleton relative to its center at any frame.
max_extent = 0
for frame in range(all_points.shape[2]):
    # Get all points for this frame
    # We need to reconstruct the points used in the skeleton lines
    # Or just use all_points if we assume they are all relevant (or close enough)
    # Using all_points is safer/easier
    frame_points = all_points[:, :, frame]
    
    # Filter out NaNs
    valid_mask = ~np.isnan(frame_points[0])
    if not np.any(valid_mask):
        continue
        
    valid_points = frame_points[:, valid_mask]
    
    center = np.mean(valid_points, axis=1)
    
    # Calculate max distance from center in any dimension
    dists = np.abs(valid_points - center[:, np.newaxis])
    max_dist = np.max(dists)
    if max_dist > max_extent:
        max_extent = max_dist

# Add a little padding
box_size = max_extent * 0.1

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

def update(frame):
    # Update lines
    current_points = []
    for line, (p1_data, p2_data) in zip(lines, lines_data):
        # p1_data shape: (3, num_frames)
        xs = [p1_data[0, frame], p2_data[0, frame]]
        ys = [p1_data[1, frame], p2_data[1, frame]]
        zs = [p1_data[2, frame], p2_data[2, frame]]
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
        
        # Collect points to calculate center for this frame
        current_points.append([p1_data[0, frame], p1_data[1, frame], p1_data[2, frame]])
        current_points.append([p2_data[0, frame], p2_data[1, frame], p2_data[2, frame]])
    
    # Update camera center
    if current_points:
        current_points = np.array(current_points)
        # Handle NaNs if any point is missing
        valid_mask = ~np.isnan(current_points[:, 0])
        if np.any(valid_mask):
            valid_points = current_points[valid_mask]
            center = np.mean(valid_points, axis=0)
            
            ax.set_xlim(center[0] - box_size, center[0] + box_size)
            ax.set_ylim(center[1] - box_size, center[1] + box_size)
            ax.set_zlim(center[2] - box_size, center[2] + box_size)

    return lines

# Create animation
num_frames = all_points.shape[2]
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

plt.show()
```

This visualization confirms the data integrity and skeleton connectivity.
 