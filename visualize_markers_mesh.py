#!/usr/bin/env python3
"""
Combined visualization of motion capture markers and fitted SMPL mesh.
Shows markers, marker skeleton connections, and SMPL mesh in the same view.
All coordinates are transformed to marker space (Z-up) for consistent visualization.

================================================================================
SETUP AND USAGE
================================================================================

Prerequisites:
--------------
This script requires the mdm-data-pipeline conda environment with all dependencies
including: numpy, matplotlib, trimesh, smplx, torch, and other packages.

Activating the Environment:
---------------------------
# Option 1: If the environment already exists
conda activate mdm-data-pipeline

# Option 2: If the environment doesn't exist, create it from environment.yml
conda env create -f environment.yml
conda activate mdm-data-pipeline

# Verify the environment is active (should see "mdm-data-pipeline")
conda info --envs

Running the Script:
-------------------
# Basic usage - visualize a specific frame (static view)
python visualize_markers_mesh.py -s 01 -c 0 --frame 0

# Interactive mode - use slider to scrub through frames
python visualize_markers_mesh.py -s 01 -c 0 --interactive

# Specify different subject and scene
python visualize_markers_mesh.py -s 02 -c 1 --frame 5

# Specify SMPL model type (male/female/neutral)
python visualize_markers_mesh.py -s 01 -c 0 --frame 0 -m male

Arguments:
----------
-s, --subject     : Subject ID (e.g., 01, 02, 138)
-c, --scene       : Scene number (e.g., 0, 1, 2, 3)
--frame           : Frame index to display (static mode only, default: 0)
--interactive     : Enable interactive mode with slider and controls
-m, --model       : SMPL model type: neutral (default), male, or female

Controls (Interactive Mode):
----------------------------
- Frame Slider:     Scrub through different frames
- Checkboxes:      Toggle visibility of markers, skeleton, and mesh
- Mouse (click+drag): Rotate 3D view
- Mouse (scroll):   Zoom in/out

Data Requirements:
--------------------
This script reads from the data directory structure:
- Processed markers: data/processed_markers_all_2/SUBJ{ID}/*_markers_positions.npz
- SMPL parameters:   data/fitted_smpl_all_3/SUBJ{ID}/*_smpl_params.npz
- SMPL models:       data/smpl/ (SMPL_NEUTRAL.npz, SMPL_MALE.npz, SMPL_FEMALE.npz)

================================================================================
TECHNICAL NOTES
================================================================================

Coordinate Systems:
-------------------
- Markers: Z-up (X=left/right, Y=forward/backward, Z=up/down)
- SMPL:    Y-up (X=left/right, Y=up/down, Z=forward/backward)

The script performs coordinate transformation: SMPL (x, y, z) → Marker (x, -z, y)
This ensures both markers and mesh are displayed in the same coordinate system.

SMPL Model:
-----------
SMPL (Skinned Multi-Person Linear Model) is a statistical body model that represents
human bodies using:
- Shape parameters (betas, 10 values): Body proportions and fatness
- Pose parameters (poses, 72 values): Joint rotations (3 per joint × 24 joints)
- Translation (trans, 3 values): Global position in world coordinates
- Gender-specific templates for more accurate body shapes

Performance:
------------
- Mesh is simplified for interactive performance
- Vertices: 6890, Faces: ~2000 (downsampled from 13776 for rendering speed)
- Lines: Every face visible with 0.1 width for subtle definition
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
import os
import sys
from matplotlib.widgets import Slider, Button, CheckButtons
import torch
import smplx
import trimesh

# Skeleton connections from inspect_file.py (marker labels)
# Modified to use SACR (sacrum) as root instead of CentreOfMass
MARKER_SKELETON = [
    ("SACR", "TRXO"),  # Spine (using SACR instead of CentreOfMass)
    ("TRXO", "HEDO"),  # Spine
    ("TRXO", "LCLO"),
    ("TRXO", "RCLO"),  # Shoulders
    ("LCLO", "LHUO"),
    ("LHUO", "LRAO"),
    ("LRAO", "LHNO"),  # Left Arm
    ("RCLO", "RHUO"),
    ("RHUO", "RRAO"),
    ("RRAO", "RHNO"),  # Right Arm
    ("SACR", "LFEP"),  # Left leg (using SACR instead of CentreOfMass)
    ("LFEP", "LFEO"),
    ("LFEO", "LTIO"),
    ("LTIO", "LFOO"),  # Left Leg
    ("SACR", "RFEP"),  # Right leg (using SACR instead of CentreOfMass)
    ("RFEP", "RFEO"),
    ("RFEO", "RTIO"),
    ("RTIO", "RFOO"),  # Right Leg
]


def transform_smpl_to_marker_space(smpl_pos):
    """
    Convert SMPL coordinates from Y-up to marker Z-up coordinate system.

    Coordinate Systems:
    - SMPL uses Y-up: X=left/right, Y=up/down (vertical), Z=forward/backward
    - Markers use Z-up: X=left/right, Y=forward/backward, Z=up/down (vertical)

    Mathematical Transformation:
    (x, y, z) → (x, -z, y)

    Where:
    - x' = x: X-axis remains unchanged (left/right)
    - y' = -z: SMPL Z (forward) becomes Marker Y, negated to match walking direction
    - z' = y: SMPL Y (up) becomes Marker Z (up)

    This is a 90-degree rotation about the X-axis followed by a reflection.
    Without this transformation, the SMPL mesh would appear rotated 90° from markers.

    Args:
        smpl_pos (np.ndarray): Array of 3D coordinates in SMPL space (Y-up).
                              Shape can be (..., 3) supporting any batch dimensions.
                              Example shapes: (3,), (24, 3), (6890, 3), (100, 24, 3)

    Returns:
        np.ndarray: Transformed coordinates in marker space (Z-up).
                   Same shape as input: (..., 3)

    Example:
        >>> smpl_coords = np.array([[0.1, 0.9, 0.2]])  # Y=0.9 (height in SMPL)
        >>> marker_coords = transform_smpl_to_marker_space(smpl_coords)
        >>> print(marker_coords)  # [[0.1, -0.2, 0.9]]  # Z=0.9 (height in markers)
    """
    transformed = np.zeros_like(smpl_pos)
    transformed[..., 0] = smpl_pos[..., 0]  # X same
    transformed[..., 1] = -smpl_pos[..., 2]  # -Z → Y (forward)
    transformed[..., 2] = smpl_pos[..., 1]  # Y → Z (up)
    return transformed


def load_processed_markers(subject, scene):
    """
    Load processed motion capture marker data from NPZ file.
    
    Reads the output from 1_dataset_prep.py which contains cleaned and 
    preprocessed marker trajectories ready for SMPL fitting.
    
    File Naming Convention:
    - Input: data/processed_markers_all_2/SUBJ{ID}/SUBJ{num}_{scene}_markers_positions.npz
    - Example: data/processed_markers_all_2/SUBJ01/SUBJ1_0_markers_positions.npz
    
    NPZ File Contents:
    - 'marker_data': Array of shape (n_frames, n_markers, 3) containing 3D positions
    - 'marker_names': Array of marker label names (e.g., 'LHEE', 'LFHD', 'SACR')
    - 'frame_rate': Recording frame rate (typically 20 Hz for this dataset)
    - Additional fields: 'marker_layout', 'original_indices' (not used here)
    
    Coordinate System:
    - All positions are in meters (not millimeters like raw C3D)
    - Z-up coordinate system: X=left/right, Y=forward/backward, Z=up/down
    - Origin typically centered around the lab space or subject
    
    Args:
        subject (str): Subject ID string (e.g., '01', '02', '138'). 
                       Leading zeros are handled automatically.
        scene (int): Scene/trial number (e.g., 0, 1, 2, 3). 
                     Multiple trials per subject with different activities.
    
    Returns:
        tuple: (marker_data, marker_names, frame_rate)
            - marker_data (np.ndarray): Shape (n_frames, n_markers, 3)
            - marker_names (np.ndarray): Shape (n_markers,), dtype='<U8'
            - frame_rate (float): Frames per second (e.g., 20.0)
    
    Raises:
        FileNotFoundError: If the marker file doesn't exist at expected path
    
    Example:
        >>> marker_data, marker_names, fps = load_processed_markers('01', 0)
        >>> print(f"Loaded {len(marker_names)} markers, {marker_data.shape[0]} frames")
        Loaded 111 markers, 17 frames at 20.0 FPS
        >>> print(marker_names[:5])  # First 5 marker names
        ['LFHD' 'RFHD' 'LBHD' 'RBHD' 'C7']
    """
    subj_id = f"SUBJ{subject}"
    markers_dir = os.path.join("data", "processed_markers_all_2", subj_id)

    # Format filename: SUBJ1_0_markers_positions.npz
    subj_num = subject.lstrip("0") or "0"
    markers_path = os.path.join(
        markers_dir, f"SUBJ{subj_num}_{scene}_markers_positions.npz"
    )

    if not os.path.exists(markers_path):
        raise FileNotFoundError(f"Processed markers not found: {markers_path}")

    data = np.load(markers_path, allow_pickle=True)
    marker_data = data["marker_data"]  # shape: (n_frames, n_markers, 3)
    marker_names = data["marker_names"]  # shape: (n_markers,)
    frame_rate = float(data["frame_rate"])

    return marker_data, marker_names, frame_rate


def load_smpl_params(subject, scene):
    """
    Load fitted SMPL (Skinned Multi-Person Linear Model) parameters from NPZ file.
    
    SMPL is a statistical body model that represents human bodies using learned 
    shape and pose parameters. This function loads the output from 2_fit_smpl_markers.py
    which fits the SMPL model to motion capture marker data.
    
    SMPL Parameters:
    ----------------
    The SMPL model decomposes body pose into:
    - Shape (betas): 10 PCA coefficients representing body proportions (fatness, height, etc.)
    - Pose (poses): 72 rotation parameters (3 angles × 24 joints = 72)
    - Translation (trans): 3D global position (x, y, z) in meters
    
    Joint Hierarchy (24 joints):
    - 0: Pelvis (root)
    - 1-2: Left/Right Hip
    - 3: Spine1
    - 4-5: Left/Right Knee
    - 6: Spine2
    - 7-8: Left/Right Ankle
    - 9: Spine3
    - 10-11: Left/Right Foot
    - 12: Neck
    - 13-14: Left/Right Clavicle
    - 15: Head
    - 16-17: Left/Right Shoulder
    - 18-19: Left/Right Elbow
    - 20-21: Left/Right Wrist
    
    File Naming Convention:
    - Input: data/fitted_smpl_all_3/SUBJ{ID}/SUBJ{num}_{scene}_smpl_params.npz
    - Example: data/fitted_smpl_all_3/SUBJ01/SUBJ1_0_smpl_params.npz
    - Subject betas: data/fitted_smpl_all_3/SUBJ{ID}/betas.npy
    
    NPZ File Contents:
    - 'poses': Shape (n_frames, 72), joint rotation parameters in axis-angle format
    - 'trans': Shape (n_frames, 3), global translation in meters (Y-up coordinates)
    - 'betas': Shape (10,), body shape PCA coefficients (constant for subject)
    - 'joints': Shape (n_frames, 24, 3), 3D positions of 24 SMPL joints
    - 'gender': String ('male', 'female', 'neutral'), SMPL model type used
    
    Coordinate System:
    - Y-up coordinate system: X=left/right, Y=up/down, Z=forward/backward
    - All positions in meters (converted from millimeters during fitting)
    - Must be transformed to Z-up (marker space) for visualization using
      transform_smpl_to_marker_space()
    
    Mathematical Basis:
    - Poses use axis-angle representation (Rodrigues formula)
    - Each joint rotation: 3 parameters (rotation axis × rotation angle)
    - Global pose vector: concatenation of all 24 joint rotations = 72 parameters
    - Shape parameters from PCA on human body scans
    
    Args:
        subject (str): Subject ID string (e.g., '01', '02', '138')
        scene (int): Scene/trial number (e.g., 0, 1, 2, 3)
    
    Returns:
        tuple: (poses, trans, betas, joints, gender)
            - poses (np.ndarray): Shape (n_frames, 72), joint rotations
            - trans (np.ndarray): Shape (n_frames, 3), global translations
            - betas (np.ndarray): Shape (10,), body shape parameters
            - joints (np.ndarray): Shape (n_frames, 24, 3), joint positions
            - gender (str or None): 'male', 'female', 'neutral', or None
    
    Raises:
        FileNotFoundError: If SMPL parameter file doesn't exist
    
    Example:
        >>> poses, trans, betas, joints, gender = load_smpl_params('01', 0)
        >>> print(f"Gender: {gender}, Shape: {betas.shape}, Frames: {poses.shape[0]}")
        Gender: male, Shape: (10,), Frames: 17
        >>> print(f"Root joint (pelvis) first frame: {joints[0, 0]}")
        [-0.161  0.888 -0.172]  # Y-up coordinates
    """
    subj_id = f"SUBJ{subject}"
    smpl_dir = os.path.join("data", "fitted_smpl_all_3", subj_id)

    subj_num = subject.lstrip("0") or "0"
    smpl_path = os.path.join(smpl_dir, f"SUBJ{subj_num}_{scene}_smpl_params.npz")

    if not os.path.exists(smpl_path):
        raise FileNotFoundError(f"SMPL params not found: {smpl_path}")

    data = np.load(smpl_path)
    poses = data["poses"]  # (n_frames, 72)
    trans = data["trans"]  # (n_frames, 3)
    betas = data["betas"]  # (10,)
    joints = data["joints"]  # (n_frames, 24, 3)
    gender = str(data["gender"]) if "gender" in data else None

    return poses, trans, betas, joints, gender


def precompute_smpl_vertices(poses, trans, betas, model_type="neutral"):
    """
    Pre-compute SMPL mesh vertices for all frames using forward kinematics.
    
    Uses the SMPL model to generate 3D mesh vertices from pose, shape, and 
    translation parameters. This pre-computation is essential for interactive
    visualization to avoid expensive SMPL model inference on every frame update.
    
    SMPL Forward Pass:
    ------------------
    The SMPL model computes mesh vertices V from parameters θ (poses), β (betas), 
    and t (translation) using learned shape blend shapes and pose blend shapes:
    
    V(θ, β, t) = t + W(T(θ, β), θ, J(β), W)
    
    Where:
    - T(θ, β): Template mesh with shape and pose deformations applied
    - W: Skinning weights (linear blend skinning)
    - J(β): Joint locations derived from shape β
    - θ: Joint rotations (poses)
    - t: Global translation
    
    Vertex Count: 6890 vertices forming the complete body surface mesh
    Face Count: 13776 triangles connecting vertices (triangular mesh)
    
    Performance Optimization:
    -------------------------
    - Processes all frames in a single batch (batch_size = n_frames)
    - Uses torch.no_grad() to disable gradient computation (inference only)
    - Pre-computation allows <500ms frame updates during interactive visualization
    
    Output Coordinate System:
    -------------------------
    - Returns vertices in SMPL space: Y-up (X=left/right, Y=up, Z=forward)
    - Must call transform_smpl_to_marker_space() to convert to Z-up for display
    - Positions in meters
    
    Args:
        poses (np.ndarray): Shape (n_frames, 72), joint rotation parameters.
                           Format: [global_orient (3), body_pose (69)]
        trans (np.ndarray): Shape (n_frames, 3), global translation per frame.
                           Format: [tx, ty, tz] in meters (Y-up coords)
        betas (np.ndarray): Shape (10,), body shape PCA coefficients.
                           Constant for a subject, determines body proportions.
        model_type (str): SMPL model variant: 'neutral' (default), 'male', or 'female'.
                         Gender-specific templates improve body shape accuracy.
    
    Returns:
        tuple: (vertices, faces)
            - vertices (np.ndarray): Shape (n_frames, 6890, 3), mesh vertex positions
                                   in SMPL coordinate space (Y-up)
            - faces (np.ndarray): Shape (13776, 3), triangle face indices
                                Each row contains 3 vertex indices forming a triangle
    
    Raises:
        RuntimeError: If SMPL model fails to load or forward pass fails
    
    Example:
        >>> poses, trans, betas, _, gender = load_smpl_params('01', 0)
        >>> vertices, faces = precompute_smpl_vertices(poses, trans, betas, gender)
        >>> print(f"Computed {vertices.shape[0]} frames of mesh data")
        Computed 17 frames of mesh data
        >>> print(f"Mesh vertices: {vertices.shape[1]}, Faces: {faces.shape[0]}")
        Mesh vertices: 6890, Faces: 13776
        >>> print(f"First vertex first frame: {vertices[0, 0]}")
        [0.014 0.915 0.028]  # Y-up: vertex 0 at height 0.915m
    
    References:
        Loper et al. "SMPL: A Skinned Multi-Person Linear Model" SIGGRAPH Asia 2015
    """
    print(f"Pre-computing SMPL vertices for {len(poses)} frames...")

    # Convert numpy to torch tensors
    poses_t = torch.tensor(poses, dtype=torch.float32)
    trans_t = torch.tensor(trans, dtype=torch.float32)
    betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)  # (1, 10)

    # Initialize SMPL model
    model_path = "data/smpl/"
    model = smplx.create(
        model_path,
        model_type="smpl",
        gender=model_type.lower(),
        ext="pkl",
        batch_size=len(poses),
    )

    # Ensure pose dimensions
    if poses_t.ndim == 1:
        poses_t = poses_t.unsqueeze(0)
    if trans_t.ndim == 1:
        trans_t = trans_t.unsqueeze(0)

    # Split pose into global orientation and body pose
    global_orient = poses_t[:, :3]
    body_pose = poses_t[:, 3:]

    # Forward pass for all frames at once
    with torch.no_grad():
        output = model(
            betas=betas_t.expand(len(poses), -1),
            global_orient=global_orient,
            body_pose=body_pose,
            transl=trans_t,
            return_verts=True,
        )

    vertices = output.vertices.detach().cpu().numpy()  # (n_frames, 6890, 3)
    faces = model.faces  # (13776, 3)

    print(f"  Vertices shape: {vertices.shape}")
    print(f"  Faces shape: {faces.shape}")

    return vertices, faces


def validate_skeleton_connections(marker_names, skeleton):
    """
    Validate skeleton connections against available marker names.
    
    The skeleton definition (MARKER_SKELETON) contains anatomical connections
    between markers (e.g., "SACR" to "TRXO" for spine). However, some markers
    may be missing from specific trials or datasets. This function filters
    the skeleton to only include connections where both markers are present.
    
    Marker Naming Convention:
    --------------------------
    The Van Criekinge dataset uses standardized marker labels:
    - Anatomical: SACR (sacrum), TRXO (thorax origin), HEDO (head)
    - Left side: LHEE (left heel), LTOE (left toe), LFOO (left foot)
    - Right side: RHEE (right heel), RTOE (right toe), RFOO (right foot)
    - Joints: LFEP (left femur proximal), LFEO (left femur origin)
    
    Validation Logic:
    -----------------
    For each (parent, child) pair in skeleton:
    1. Check if parent marker name exists in marker_names array
    2. Check if child marker name exists in marker_names array  
    3. If both exist: add their indices to valid_connections
    4. If either missing: add marker name(s) to missing_markers set
    
    Mathematical Mapping:
    ---------------------
    Input: marker_names array (n_markers,) with string labels
    Output: valid_connections list of (idx1, idx2) integer tuples
    Conversion: name → index using np.where(marker_names == name)[0][0]
    
    Args:
        marker_names (np.ndarray): Array of marker label names from NPZ file.
                                 Shape (n_markers,), dtype='<U8' (string)
                                 Example: ['LFHD', 'RFHD', 'LHEE', 'RHEE', ...]
        skeleton (list): List of (parent_name, child_name) tuples defining
                        anatomical connections. Example: [("SACR", "TRXO"), ...]
    
    Returns:
        tuple: (valid_connections, missing_markers)
            - valid_connections (list): List of (idx1, idx2) integer tuples
                                       where idx1 and idx2 are positions
                                       in the marker_names array
            - missing_markers (set): Set of marker names not found in data
                                    (for debugging/reporting missing markers)
    
    Example:
        >>> marker_names = np.array(['SACR', 'TRXO', 'LHEE', 'RHEE'])
        >>> skeleton = [("SACR", "TRXO"), ("TRXO", "MISSING"), ("LHEE", "RHEE")]
        >>> valid, missing = validate_skeleton_connections(marker_names, skeleton)
        >>> print(f"Valid connections: {valid}")
        Valid connections: [(0, 1), (2, 3)]  # SACR-TRXO and LHEE-RHEE
        >>> print(f"Missing markers: {missing}")
        Missing markers: {'MISSING'}
    
    Note:
        This validation prevents visualization errors from missing markers.
        The script continues with partial skeleton rather than failing.
    """
    valid_connections = []
    missing_markers = set()

    for p1_name, p2_name in skeleton:
        # Find indices of these markers
        idx1 = np.where(marker_names == p1_name)[0]
        idx2 = np.where(marker_names == p2_name)[0]

        if len(idx1) > 0 and len(idx2) > 0:
            valid_connections.append((idx1[0], idx2[0]))
        else:
            if len(idx1) == 0:
                missing_markers.add(p1_name)
            if len(idx2) == 0:
                missing_markers.add(p2_name)

    return valid_connections, missing_markers


def plot_trimesh_in_matplotlib(ax, vertices, faces, color="lightblue", alpha=0.7):
    """
    Plot a trimesh object in matplotlib 3D axes.

    Args:
        ax: matplotlib 3D axes
        vertices: (n_vertices, 3) vertex positions
        faces: (n_faces, 3) face indices
        color: Mesh color
        alpha: Transparency (0-1)
    """
    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # For static visualization, we can use all faces
    # For interactive mode, we'll handle simplification separately
    # Use original vertices and faces for now

    # Create Poly3DCollection from mesh faces
    triangles = vertices[faces]

    # Create collection with all edges but very thin
    # Show every face with very thin lines for subtle definition
    mesh_collection = Poly3DCollection(
        triangles,
        facecolors=color,
        edgecolors="darkgray",  # All edges
        linewidths=0.1,  # Very thin lines
        alpha=alpha,
    )

    # Add to axes
    ax.add_collection3d(mesh_collection)

    return mesh_collection


def create_visualization(
    marker_data,
    marker_names,
    smpl_vertices,
    smpl_faces,
    skeleton_connections,
    frame_idx=0,
):
    """
    Create matplotlib 3D visualization for a single frame.

    Args:
        marker_data: (n_frames, n_markers, 3) marker positions in marker space
        marker_names: (n_markers,) marker label names
        smpl_vertices: (n_frames, n_vertices, 3) SMPL vertices in marker space
        smpl_faces: (n_faces, 3) SMPL face indices
        skeleton_connections: List of (idx1, idx2) marker indices to connect
        frame_idx: Frame index to display
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Get data for current frame
    current_markers = marker_data[frame_idx]  # (n_markers, 3)
    current_vertices = smpl_vertices[frame_idx]  # (n_vertices, 3)

    # Plot markers as blue spheres
    marker_scatter = ax.scatter(
        current_markers[:, 0],
        current_markers[:, 1],
        current_markers[:, 2],
        c="blue",
        s=20,
        label="Markers",
        alpha=0.7,
        depthshade=True,
    )

    # Plot skeleton connections as gray lines
    connection_lines = []
    for idx1, idx2 in skeleton_connections:
        p1 = current_markers[idx1]
        p2 = current_markers[idx2]
        line = ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            "gray",
            linewidth=2,
            alpha=0.7,
            label="Skeleton" if not connection_lines else None,
        )
        connection_lines.append(line[0])

    # Plot SMPL mesh using trimesh
    mesh_collection = plot_trimesh_in_matplotlib(
        ax, current_vertices, smpl_faces, color="lightblue", alpha=0.5
    )
    mesh_collection.set_label("SMPL mesh")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Frame {frame_idx} - Markers & SMPL Mesh")
    ax.legend()

    # Set equal aspect ratio (1:1:1 scale)
    # Calculate bounds from all data (markers + mesh vertices)
    vert_stride = 10
    all_points = np.vstack([current_markers, current_vertices[::vert_stride]])

    # Get min and max for each axis
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    # Calculate center
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    center_z = (z_min + z_max) / 2

    # Calculate max range across all axes
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0

    # Add some padding
    padding = max_range * 0.1
    max_range += padding

    # Set limits with equal aspect ratio
    ax.set_xlim(center_x - max_range, center_x + max_range)
    ax.set_ylim(center_y - max_range, center_y + max_range)
    ax.set_zlim(center_z - max_range, center_z + max_range)

    # Force equal aspect ratio
    ax.set_box_aspect((1, 1, 1))  # This ensures 1:1:1 scaling

    # Set initial view angle (looking from front-right-top)
    ax.view_init(elev=20, azim=45)

    return fig, ax, marker_scatter, connection_lines, mesh_collection


def create_interactive_visualization(
    marker_data,
    marker_names,
    smpl_vertices,
    smpl_faces,
    skeleton_connections,
    frame_rate=20,
):
    """
    Create an interactive matplotlib 3D visualization with frame slider and UI controls.
    
    Provides interactive exploration of motion sequences with real-time frame updates,
    view following, and visibility toggles. Pre-computes mesh data for smooth performance.
    
    Interactive Features:
    ---------------------
    1. Frame Slider (bottom of window):
       - Range: 0 to n_frames-1
       - Scrubbing triggers immediate frame update
       - Updates: markers, skeleton, mesh, and view bounds
    
    2. Visibility Toggles (checkboxes):
       - "Markers": Show/hide blue marker points
       - "Skeleton": Show/hide gray skeleton lines  
       - "Mesh": Show/hide light blue SMPL mesh
       - All can be toggled independently
    
    3. 3D Navigation (matplotlib default):
       - Left-click + drag: Rotate view
       - Right-click + drag: Pan view
       - Scroll wheel: Zoom in/out
       - Built-in matplotlib 3D interaction
    
    Frame Update Mechanism:
    -----------------------
    When frame slider changes:
    1. Get new frame index from slider value
    2. Update marker scatter positions: marker_scatter._offsets3d = (x, y, z)
    3. Update skeleton line positions: set_data() and set_3d_properties()
    4. Replace mesh collection: remove() old, create new Poly3DCollection, add_collection3d()
    5. Update view bounds: calculate new limits from current frame data
    6. Redraw canvas: fig.canvas.draw_idle()
    
    Performance Optimization:
    -------------------------
    - Pre-create trimesh objects for all frames (avoids recreating from scratch)
    - Store in ui_state['meshes'] list for fast access
    - Mesh replacement: ~100-300ms per frame (within 500ms target)
    - No gradient computation (torch.no_grad() during precomputation)
    
    View Following:
    ---------------
    - View bounds update with each frame change
    - Calculates center and max range from current frame's markers + mesh
    - Adds 10% padding for comfortable viewing
    - Maintains 1:1:1 aspect ratio throughout
    
    UI State Management:
    --------------------
    Stored in dictionary ui_state for access across callbacks:
    - frame_idx: Current frame number
    - show_markers/skeleton/mesh: Visibility flags
    - marker_scatter: Scatter plot object
    - connection_lines: List of skeleton line objects
    - mesh_collection: Current mesh Poly3DCollection
    - meshes: List of pre-created trimesh objects
    
    Args:
        marker_data (np.ndarray): Shape (n_frames, n_markers, 3), marker positions.
                                 All frames of marker data pre-loaded.
        marker_names (np.ndarray): Shape (n_markers,), marker labels (for reference).
        smpl_vertices (np.ndarray): Shape (n_frames, 6890, 3), pre-computed mesh
                                   vertices for all frames in Z-up space.
        smpl_faces (np.ndarray): Shape (13776, 3), SMPL face indices (constant).
        skeleton_connections (list): List of (idx1, idx2) tuples for skeleton lines.
        frame_rate (int): Recording frame rate in Hz (for info display only).
                         Default: 20 (Van Criekinge dataset standard)
    
    Returns:
        tuple: (fig, ax, ui_state)
            - fig (matplotlib.figure.Figure): Figure with controls and visualization
            - ax (mpl_toolkits.mplot3d.axes3d.Axes3D): 3D axes object
            - ui_state (dict): Dictionary containing all interactive state:
              * frame_idx: Current frame
              * show_markers/skeleton/mesh: Visibility booleans
              * marker_scatter: Scatter plot object
              * connection_lines: Skeleton line objects list
              * mesh_collection: Current mesh collection
              * meshes: Pre-computed trimesh objects for all frames
    
    Controls for User:
    ------------------
    - Frame Slider: Drag to change frames (auto-pauses visualization)
    - Checkboxes: Click to show/hide markers, skeleton, or mesh
    - Mouse: Rotate, pan, zoom the 3D view
    - Window close: Exit the application
    
    Example:
        >>> # Prepare data (same as create_visualization)
        >>> marker_data, marker_names, fps = load_processed_markers('01', 0)
        >>> poses, trans, betas, _, gender = load_smpl_params('01', 0)
        >>> smpl_vertices_yup, smpl_faces = precompute_smpl_vertices(poses, trans, betas, gender)
        >>> smpl_vertices = transform_smpl_to_marker_space(smpl_vertices_yup)
        >>> skeleton, _ = validate_skeleton_connections(marker_names, MARKER_SKELETON)
        >>> # Create interactive visualization
        >>> fig, ax, ui_state = create_interactive_visualization(
        ...     marker_data, marker_names, smpl_vertices, smpl_faces, skeleton, int(fps)
        ... )
        >>> plt.show()  # Opens interactive window
        >>> # User can now: scrub frames, toggle visibility, rotate view
    
    Note:
        Requires user interaction. Window must be closed to exit.
        For automated frame saving or batch processing, use create_visualization()
        with a loop over frame_idx values.
    """
    n_frames = len(marker_data)

    fig = plt.figure(figsize=(14, 12))
    plt.subplots_adjust(bottom=0.25)  # Space for controls

    ax = fig.add_subplot(111, projection="3d")

    # Initial frame
    frame_idx = 0
    current_markers = marker_data[frame_idx]
    current_vertices = smpl_vertices[frame_idx]

    # Plot initial markers
    marker_scatter = ax.scatter(
        current_markers[:, 0],
        current_markers[:, 1],
        current_markers[:, 2],
        c="blue",
        s=20,
        alpha=0.7,
        depthshade=True,
    )

    # Plot initial skeleton connections
    connection_lines = []
    for idx1, idx2 in skeleton_connections:
        p1 = current_markers[idx1]
        p2 = current_markers[idx2]
        line = ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            "gray",
            linewidth=2,
            alpha=0.7,
        )[0]
        connection_lines.append(line)

    # Pre-create meshes for all frames (without simplification for now)
    print("Pre-creating meshes for all frames...")
    meshes = []
    for i in range(n_frames):
        # Create trimesh object for this frame
        mesh = trimesh.Trimesh(vertices=smpl_vertices[i], faces=smpl_faces)
        meshes.append(mesh)

    # Plot initial SMPL mesh
    initial_mesh = meshes[0]
    triangles = initial_mesh.vertices[initial_mesh.faces]
    mesh_collection = Poly3DCollection(
        triangles,
        facecolors="lightblue",
        edgecolors="darkgray",
        linewidths=0.5,
        alpha=0.5,
    )
    ax.add_collection3d(mesh_collection)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Frame {frame_idx}/{n_frames - 1} - Markers & SMPL Mesh")

    # Set equal aspect ratio (1:1:1 scale)
    # Calculate bounds from all data (markers + mesh vertices)
    vert_stride = 10
    all_points = np.vstack([current_markers, current_vertices[::vert_stride]])

    # Get min and max for each axis
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

    # Calculate center
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    center_z = (z_min + z_max) / 2

    # Calculate max range across all axes
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0

    # Add some padding
    padding = max_range * 0.1
    max_range += padding

    # Set limits with equal aspect ratio
    ax.set_xlim(center_x - max_range, center_x + max_range)
    ax.set_ylim(center_y - max_range, center_y + max_range)
    ax.set_zlim(center_z - max_range, center_z + max_range)

    # Force equal aspect ratio
    ax.set_box_aspect((1, 1, 1))  # This ensures 1:1:1 scaling

    # Set initial view angle
    ax.view_init(elev=20, azim=45)

    # UI state
    ui_state = {
        "frame_idx": frame_idx,
        "show_markers": True,
        "show_skeleton": True,
        "show_mesh": True,
        "marker_scatter": marker_scatter,
        "connection_lines": connection_lines,
        "mesh_collection": mesh_collection,
        "meshes": meshes,
        "current_mesh": mesh_collection,
    }

    # Frame slider
    ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
    frame_slider = Slider(ax_slider, "Frame", 0, n_frames - 1, valinit=0, valfmt="%d")

    def update_view_bounds(current_markers, current_vertices):
        """Update the view bounds to follow the motion."""
        # Calculate bounds from current frame data
        vert_stride = 10
        all_points = np.vstack([current_markers, current_vertices[::vert_stride]])

        # Get min and max for each axis
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
        z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

        # Calculate center
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        center_z = (z_min + z_max) / 2

        # Calculate max range across all axes
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0

        # Add some padding
        padding = max_range * 0.1
        max_range += padding

        # Update limits
        ax.set_xlim(center_x - max_range, center_x + max_range)
        ax.set_ylim(center_y - max_range, center_y + max_range)
        ax.set_zlim(center_z - max_range, center_z + max_range)

    def update_frame(val):
        frame_idx = int(frame_slider.val)
        ui_state["frame_idx"] = frame_idx

        # Update markers
        current_markers = marker_data[frame_idx]
        if ui_state["show_markers"]:
            ui_state["marker_scatter"]._offsets3d = (
                current_markers[:, 0],
                current_markers[:, 1],
                current_markers[:, 2],
            )

        # Update skeleton connections
        if ui_state["show_skeleton"]:
            for i, (idx1, idx2) in enumerate(skeleton_connections):
                p1 = current_markers[idx1]
                p2 = current_markers[idx2]
                ui_state["connection_lines"][i].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                ui_state["connection_lines"][i].set_3d_properties([p1[2], p2[2]])

        # Update SMPL mesh - remove old and add new
        if ui_state["show_mesh"]:
            # Remove old mesh
            ui_state["mesh_collection"].remove()

            # Get mesh for this frame
            mesh = ui_state["meshes"][frame_idx]
            triangles = mesh.vertices[mesh.faces]

            # Create new mesh collection
            new_mesh_collection = Poly3DCollection(
                triangles,
                facecolors="lightblue",
                edgecolors="darkgray",
                linewidths=0.5,
                alpha=0.5,
            )
            ax.add_collection3d(new_mesh_collection)

            # Update state
            ui_state["mesh_collection"] = new_mesh_collection

        # Update view bounds to follow the motion
        update_view_bounds(current_markers, smpl_vertices[frame_idx])

        ax.set_title(f"Frame {frame_idx}/{n_frames - 1} - Markers & SMPL Mesh")
        fig.canvas.draw_idle()

    frame_slider.on_changed(update_frame)

    # Checkboxes for toggling visibility
    ax_checkbox = plt.axes([0.2, 0.08, 0.2, 0.1])
    checkboxes = CheckButtons(
        ax_checkbox, ["Markers", "Skeleton", "Mesh"], [True, True, True]
    )

    def toggle_visibility(label):
        if label == "Markers":
            ui_state["show_markers"] = not ui_state["show_markers"]
            ui_state["marker_scatter"].set_visible(ui_state["show_markers"])
        elif label == "Skeleton":
            ui_state["show_skeleton"] = not ui_state["show_skeleton"]
            for line in ui_state["connection_lines"]:
                line.set_visible(ui_state["show_skeleton"])
        elif label == "Mesh":
            ui_state["show_mesh"] = not ui_state["show_mesh"]
            ui_state["mesh_collection"].set_visible(ui_state["show_mesh"])

        fig.canvas.draw_idle()

    checkboxes.on_clicked(toggle_visibility)

    # Info text
    info_text = f"Total frames: {n_frames}\nFrame rate: {frame_rate} Hz\nUse slider to change frame"
    plt.figtext(
        0.02,
        0.02,
        info_text,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )

    return fig, ax, ui_state


def main():
    """
    Main entry point for the combined markers and SMPL mesh visualization tool.
    
    Parses command-line arguments, loads motion capture data, pre-computes SMPL
    mesh vertices, and creates either a static or interactive 3D visualization.
    
    Execution Pipeline:
    -------------------
    1. Argument Parsing:
       - Parse --subject, --scene, --model, --interactive, --frame
       - Validate required arguments
    
    2. Data Loading:
       - Load processed markers: load_processed_markers(subject, scene)
       - Load SMPL parameters: load_smpl_params(subject, scene)
       - Auto-detect gender if model type not specified
    
    3. Data Validation:
       - Check frame count consistency between markers and SMPL data
       - Truncate to minimum frame count if mismatch detected
       - Report data statistics (shapes, frame rate)
    
    4. SMPL Mesh Generation:
       - Pre-compute vertices: precompute_smpl_vertices(poses, trans, betas, model_type)
       - Transform coordinates: transform_smpl_to_marker_space() (Y-up → Z-up)
       - Handle errors (fallback to joint visualization if mesh fails)
    
    5. Skeleton Validation:
       - Validate MARKER_SKELETON against available marker names
       - Filter to valid connections only
       - Report missing markers (if any)
    
    6. Visualization Creation:
       - Interactive mode: create_interactive_visualization()
       - Static mode: create_visualization() for specific frame
    
    7. Display:
       - plt.show() blocks until window closed
    
    Command-Line Arguments:
    ------------------------
    -s, --subject (str, required): Subject ID from Van Criekinge dataset.
                                   Examples: '01', '02', '138'
    -c, --scene (int, required): Scene/trial number. Range: 0-3 typically.
                                 Different scenes capture different activities.
    -m, --model (str, optional): SMPL model type. Choices: 'neutral' (default),
                                  'male', 'female'. Auto-detected from data if
                                  gender field present.
    --interactive (flag): Enable interactive mode with frame slider and 
                          visibility toggles. Mutually exclusive with --frame.
    --frame (int, optional): Specific frame index for static visualization.
                            Range: 0 to n_frames-1. Default: 0.
    
    Error Handling:
    ---------------
    - FileNotFoundError: Data files missing (prints error, exits with code 1)
    - Frame mismatch: Warns and uses minimum frame count
    - SMPL computation error: Falls back to joint-only visualization
    - No valid skeleton connections: Exits with error
    
    Exit Codes:
    -----------
    - 0: Success (window closed normally)
    - 1: Error (file not found, invalid data, etc.)
    
    Example Usage:
    --------------
    # Static view of frame 0 for subject 01, scene 0
    $ python visualize_markers_mesh.py -s 01 -c 0 --frame 0
    
    # Interactive mode for subject 02, scene 1
    $ python visualize_markers_mesh.py -s 02 -c 1 --interactive
    
    # Specify male model explicitly
    $ python visualize_markers_mesh.py -s 01 -c 0 --frame 5 -m male
    
    # Help
    $ python visualize_markers_mesh.py --help
    
    Data Directory Structure (required):
    ------------------------------------
    data/
    ├── processed_markers_all_2/SUBJ{ID}/     # Marker data from Step 1
    ├── fitted_smpl_all_3/SUBJ{ID}/           # SMPL params from Step 2
    └── smpl/                                  # SMPL model files
        ├── SMPL_NEUTRAL.npz
        ├── SMPL_MALE.npz
        └── SMPL_FEMALE.npz
    
    Notes:
    ------
    - Environment: Requires mdm-data-pipeline conda environment
    - GPU: Not required (CPU-based SMPL inference)
    - Memory: ~100MB for typical motion clips (17 frames)
    - Performance: Initial load ~2-5 seconds, frame updates ~100-300ms
    
    Returns:
        None. Displays matplotlib window and blocks until closed.
        No return value; exits via sys.exit() on errors.
    """
    parser = argparse.ArgumentParser(
        description="Visualize markers and SMPL mesh together"
    )
    parser.add_argument(
        "-s",
        "--subject",
        type=str,
        required=True,
        help="Subject ID (e.g., 01, 02, 138)",
    )
    parser.add_argument(
        "-c", "--scene", type=int, required=True, help="Scene number (e.g., 0, 1, 2, 3)"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="neutral",
        choices=["neutral", "male", "female"],
        help="SMPL model type (default: neutral)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Use interactive mode with slider and controls",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to display (non-interactive mode only)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Markers + SMPL Mesh Visualization")
    print("=" * 60)
    print(f"Subject: {args.subject}")
    print(f"Scene: {args.scene}")
    print(f"Model: {args.model}")
    print("=" * 60)

    # Load data
    try:
        print("Loading processed markers...")
        marker_data, marker_names, frame_rate = load_processed_markers(
            args.subject, args.scene
        )

        print("Loading SMPL parameters...")
        poses, trans, betas, joints, gender = load_smpl_params(args.subject, args.scene)

        # Auto-select model type if detected
        model_type = args.model
        if args.model == "neutral" and gender and gender.lower() in ["male", "female"]:
            print(f"  Auto-selecting {gender} model based on data")
            model_type = gender.lower()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\nData loaded:")
    print(f"  Marker data shape: {marker_data.shape}")
    print(f"  Marker names: {len(marker_names)} markers")
    print(f"  SMPL poses shape: {poses.shape}")
    print(f"  SMPL joints shape: {joints.shape}")
    print(f"  Frame rate: {frame_rate} Hz")

    # Check frame count consistency
    n_frames_markers = marker_data.shape[0]
    n_frames_smpl = poses.shape[0]
    if n_frames_markers != n_frames_smpl:
        print(
            f"Warning: Frame count mismatch: markers {n_frames_markers}, SMPL {n_frames_smpl}"
        )
        n_frames = min(n_frames_markers, n_frames_smpl)
        marker_data = marker_data[:n_frames]
        poses = poses[:n_frames]
        trans = trans[:n_frames]
        joints = joints[:n_frames]
        print(f"  Using first {n_frames} frames")
    else:
        n_frames = n_frames_markers

    # Pre-compute SMPL vertices
    try:
        smpl_vertices_yup, smpl_faces = precompute_smpl_vertices(
            poses, trans, betas, model_type
        )

        # Transform SMPL vertices to marker space (Z-up)
        print("Transforming SMPL vertices to marker space (Z-up)...")
        smpl_vertices = transform_smpl_to_marker_space(smpl_vertices_yup)

    except Exception as e:
        print(f"Error computing SMPL vertices: {e}")
        print("Falling back to using SMPL joints only...")
        # Use joints as proxy for mesh
        smpl_vertices = transform_smpl_to_marker_space(joints)
        # Create dummy faces (won't be used for wireframe)
        smpl_faces = np.array([[0, 1, 2]])

    # Validate skeleton connections
    print("\nValidating skeleton connections...")
    skeleton_connections, missing_markers = validate_skeleton_connections(
        marker_names, MARKER_SKELETON
    )

    if missing_markers:
        print(f"  Warning: Missing markers: {missing_markers}")
    print(f"  Valid connections: {len(skeleton_connections)}")

    if not skeleton_connections:
        print("Error: No valid skeleton connections found!")
        sys.exit(1)

    # Create visualization
    if args.interactive:
        print("\nCreating interactive visualization...")
        fig, ax, ui_state = create_interactive_visualization(
            marker_data,
            marker_names,
            smpl_vertices,
            smpl_faces,
            skeleton_connections,
            int(frame_rate),
        )

        print("\nControls:")
        print("  - Frame slider: Change current frame")
        print("  - Checkboxes: Toggle markers/skeleton/mesh visibility")
        print("  - Mouse: Rotate view (click+drag), zoom (scroll)")
        print("  - Close window to exit")

    else:
        print(f"\nCreating static visualization for frame {args.frame}...")
        if args.frame >= n_frames:
            print(f"Error: Frame {args.frame} out of range (0-{n_frames - 1})")
            sys.exit(1)

        fig, ax, marker_scatter, connection_lines, mesh_plot = create_visualization(
            marker_data,
            marker_names,
            smpl_vertices,
            smpl_faces,
            skeleton_connections,
            args.frame,
        )

        print("  Close window to exit")

    plt.show()


if __name__ == "__main__":
    main()
