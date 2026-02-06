#!/usr/bin/env python3
"""
GPU-accelerated visualization of motion capture markers and fitted SMPL mesh.
Uses pyrender for GPU rendering and matplotlib for interactive UI controls.

Combines the marker visualization from visualize_markers_mesh.py with the
GPU-accelerated rendering from render_smpl_mesh_live.py.

Usage:
    # Basic usage (will prompt for GPU selection if multiple detected)
    python visualize_markers_mesh.py --subject 01 --scene 0

    # Specify GPU manually
    python visualize_markers_mesh.py -s 01 -c 0 --gpu-id 1

    # Specify model type
    python visualize_markers_mesh.py -s 02 -c 1 -m male

Controls:
    Frame Slider:   Frame scrubbing (auto-pauses)
    Play/Pause:     Toggle animation playback
    Left/Right:     Rotate camera horizontally (azimuth ±15°)
    Up/Down:        Tilt camera vertically (elevation ±10°, range: -20° to 80°)
    Scroll Up:      Zoom in (decrease distance by 0.5, min: 1.5)
    Scroll Down:    Zoom out (increase distance by 0.5, max: 15.0)
    Spacebar:       Play/Pause toggle
    Checkboxes:     Toggle visibility of markers, skeleton, and mesh
"""

import os
import sys
import argparse
import subprocess
import re
from pathlib import Path
import numpy as np
import torch
import trimesh
import trimesh.transformations
import pyrender
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.animation import FuncAnimation

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


def transform_marker_to_smpl_space(marker_pos):
    """
    Convert marker coordinates from Z-up to SMPL Y-up coordinate system.

    Inverse of transform_smpl_to_marker_space() from visualize_markers_mesh.py.

    Coordinate Systems:
    - Markers use Z-up: X=left/right, Y=forward/backward, Z=up/down (vertical)
    - SMPL uses Y-up: X=left/right, Y=up/down (vertical), Z=forward/backward

    Mathematical Transformation:
    (x, y, z) → (x, z, -y)

    Where:
    - x' = x: X-axis remains unchanged (left/right)
    - y' = z: Marker Z (up) becomes SMPL Y (up)
    - z' = -y: Marker Y (forward) becomes SMPL Z (forward), negated to match walking direction

    Args:
        marker_pos (np.ndarray): Array of 3D coordinates in marker space (Z-up).
                               Shape can be (..., 3) supporting any batch dimensions.

    Returns:
        np.ndarray: Transformed coordinates in SMPL space (Y-up).
                   Same shape as input: (..., 3)
    """
    transformed = np.zeros_like(marker_pos)
    transformed[..., 0] = marker_pos[..., 0]  # X same
    transformed[..., 1] = marker_pos[..., 2]  # Z → Y (up)
    transformed[..., 2] = -marker_pos[..., 1]  # -Y → Z (forward)
    return transformed


def detect_gpus():
    """
    Detect available GPUs and return their information.

    Returns:
        list: List of dicts with 'id', 'device', 'vendor', 'model' keys
    """
    gpus = []
    dri_path = Path("/dev/dri")

    if not dri_path.exists():
        return gpus

    # Find all card devices
    cards = sorted([c for c in dri_path.glob("card*") if c.name.startswith("card") and c.name[4:].isdigit()])

    for card_idx, card in enumerate(cards):
        gpu_info = {
            "id": card_idx,
            "device": str(card),
            "vendor": "Unknown",
            "model": "Unknown",
        }

        try:
            # Get device information using udevadm
            result = subprocess.run(
                ["udevadm", "info", "--query=all", f"--name={card}"],
                capture_output=True,
                text=True,
                timeout=2,
            )

            if result.returncode == 0:
                output = result.stdout

                # Extract PCI path
                pci_match = re.search(r"ID_PATH=pci-([\S]+)", output)
                if pci_match:
                    pci_path = pci_match.group(1)

                    # Get GPU info from lspci
                    lspci_result = subprocess.run(
                        ["lspci", "-v", "-s", pci_path],
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )

                    if lspci_result.returncode == 0:
                        lspci_output = lspci_result.stdout

                        # Parse vendor and model from VGA controller line
                        vga_match = re.search(r"VGA compatible controller: (.+)", lspci_output)
                        if vga_match:
                            full_name = vga_match.group(1).strip()

                            # Split vendor and model
                            if "NVIDIA" in full_name:
                                gpu_info["vendor"] = "NVIDIA"
                                gpu_info["model"] = full_name.replace("NVIDIA Corporation", "").strip()
                            elif "AMD" in full_name or "ATI" in full_name:
                                gpu_info["vendor"] = "AMD"
                                # Clean up AMD naming
                                model = re.sub(
                                    r"Advanced Micro Devices, Inc\.\s*\[AMD/ATI\]\s*",
                                    "",
                                    full_name,
                                )
                                model = re.sub(r"\s*\(prog-if.*?\).*", "", model)
                                model = re.sub(r"\s*\(rev.*?\).*", "", model)
                                gpu_info["model"] = model.strip()
                            elif "Intel" in full_name:
                                gpu_info["vendor"] = "Intel"
                                gpu_info["model"] = full_name.replace("Intel Corporation", "").strip()
                            else:
                                gpu_info["model"] = full_name

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            # If detection fails, keep default 'Unknown' values
            pass

        gpus.append(gpu_info)

    return gpus


def select_gpu(gpus):
    """
    Prompt user to select a GPU from the available options.

    Args:
        gpus: List of GPU info dicts

    Returns:
        int: Selected GPU ID
    """
    print("\n" + "=" * 60)
    print("Multiple GPUs detected. Please select one:")
    print("=" * 60)

    for gpu in gpus:
        status = ""

        # Check if GPU has working drivers by trying glxinfo
        try:
            result = subprocess.run(
                ["glxinfo"],
                capture_output=True,
                text=True,
                timeout=2,
                env={**os.environ, "DRI_PRIME": str(gpu["id"])},
            )
            if result.returncode == 0 and gpu["vendor"] in result.stdout:
                status = " [drivers OK]"
        except Exception:
            pass

        print(f"  [{gpu['id']}] {gpu['vendor']} - {gpu['model']}{status}")
        print(f"      Device: {gpu['device']}")

    print("=" * 60)

    while True:
        try:
            choice = input(f"Select GPU [0-{len(gpus) - 1}]: ").strip()
            gpu_id = int(choice)
            if 0 <= gpu_id < len(gpus):
                selected = gpus[gpu_id]
                print(f"\nSelected: {selected['vendor']} - {selected['model']}")
                return gpu_id
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(gpus) - 1}")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n\nAborted by user.")
            sys.exit(1)


def setup_rendering_backend(gpu_id=None):
    """
    Setup the EGL rendering backend with GPU selection.

    Args:
        gpu_id: GPU device ID to use (None for auto-detection)

    Returns:
        int or None: Selected GPU ID (None if using default EGL device)
    """
    # Always use EGL backend
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    # Detect available GPUs
    gpus = detect_gpus()

    if len(gpus) == 0:
        print("Warning: No GPU devices detected. Trying default EGL device...")
        return None

    if len(gpus) == 1:
        # Only one GPU, use it automatically
        gpu_id = 0
        print(f"Using GPU: {gpus[0]['vendor']} - {gpus[0]['model']}")
        os.environ["EGL_DEVICE_ID"] = str(gpu_id)
        return gpu_id

    # Multiple GPUs detected
    if gpu_id is None:
        # Prompt user to select
        gpu_id = select_gpu(gpus)
    elif gpu_id >= len(gpus):
        print(f"Error: GPU ID {gpu_id} not found. Available GPUs: 0-{len(gpus) - 1}")
        sys.exit(1)
    else:
        # Use specified GPU
        print(f"Using GPU: {gpus[gpu_id]['vendor']} - {gpus[gpu_id]['model']}")

    os.environ["EGL_DEVICE_ID"] = str(gpu_id)
    return gpu_id


def setup_renderer_live(width=1024, height=1024):
    """
    Initialize offscreen renderer for live viewing.

    Args:
        width: Render width in pixels
        height: Render height in pixels

    Returns:
        pyrender.OffscreenRenderer instance
    """
    try:
        renderer = pyrender.OffscreenRenderer(width, height)
        return renderer
    except Exception as e:
        print(f"Renderer initialization failed: {e}")
        print("\nEnsure EGL/OpenGL dependencies are installed and GPU drivers are working.")
        print("On headless servers, ensure libEGL is available.")
        sys.exit(1)


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
    """
    subj_id = f"SUBJ{subject}"
    markers_dir = os.path.join("data", "processed_markers_all_2", subj_id)

    # Format filename: SUBJ1_0_markers_positions.npz
    subj_num = subject.lstrip("0") or "0"
    markers_path = os.path.join(markers_dir, f"SUBJ{subj_num}_{scene}_markers_positions.npz")

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
    """
    subj_id = f"SUBJ{subject}"
    smpl_dir = os.path.join("data", "fitted_smpl_all_3_new", subj_id)

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
    """
    print(f"Pre-computing SMPL vertices for {len(poses)} frames...")

    # Convert numpy to torch tensors
    poses_t = torch.tensor(poses, dtype=torch.float32)
    trans_t = torch.tensor(trans, dtype=torch.float32)
    betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0)  # (1, 10)

    # Initialize SMPL model
    import smplx

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

    Args:
        marker_names (np.ndarray): Array of marker label names from NPZ file.
                                  Shape (n_markers,), dtype='<U8' (string)
        skeleton (list): List of (parent_name, child_name) tuples defining
                         anatomical connections.

    Returns:
        tuple: (valid_connections, missing_markers)
            - valid_connections (list): List of (idx1, idx2) integer tuples
            - missing_markers (set): Set of marker names not found in data
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


def create_floor_mesh():
    """Create a simple flat floor plane for visual reference."""
    grid_size = 4.0

    # Create a simple quad at y=0
    vertices = np.array(
        [
            [-grid_size, 0, -grid_size],
            [grid_size, 0, -grid_size],
            [-grid_size, 0, grid_size],
            [grid_size, 0, grid_size],
        ]
    )

    # Two triangles with CCW winding for upward normals
    faces = np.array([[0, 2, 1], [1, 2, 3]])  # First triangle  # Second triangle

    floor_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    floor_mesh.visual.vertex_colors = [150, 150, 180, 150]  # Light blue-gray

    # Material with double-sided rendering
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, roughnessFactor=0.8, alphaMode="BLEND", doubleSided=True
    )

    return pyrender.Mesh.from_trimesh(floor_mesh, material=material, smooth=False)


def create_scene_with_markers_and_mesh(
    markers,
    skeleton_connections,
    mesh_vertices,
    mesh_faces,
    show_markers=True,
    show_skeleton=True,
    show_mesh=True,
):
    """
    Create a pyrender scene with markers, skeleton lines, and SMPL mesh.

    Args:
        markers: (n_markers, 3) marker positions in SMPL space (Y-up)
        skeleton_connections: List of (idx1, idx2) marker indices to connect
        mesh_vertices: (n_vertices, 3) SMPL mesh vertices
        mesh_faces: (n_faces, 3) SMPL mesh faces
        show_markers: Whether to show marker points
        show_skeleton: Whether to show skeleton lines
        show_mesh: Whether to show SMPL mesh

    Returns:
        pyrender.Scene: Scene with all elements added
    """
    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[255, 255, 255])

    # Add SMPL mesh if visible
    if show_mesh:
        body_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces)
        body_mesh.visual.vertex_colors = [174, 199, 232, 255]  # Light blue color
        mesh_node = pyrender.Mesh.from_trimesh(body_mesh, smooth=True)
        scene.add(mesh_node)

    # Add markers as points if visible
    if show_markers and len(markers) > 0:
        # Create points with bright blue color
        colors = np.zeros((len(markers), 4))
        colors[:, 0] = 0.0  # R
        colors[:, 1] = 0.4  # G (slightly greenish for better visibility)
        colors[:, 2] = 1.0  # B (bright blue)
        colors[:, 3] = 1.0  # Alpha

        # Create mesh from points - use spheres instead of points for better visibility
        # Create small spheres at each marker position
        for i, point in enumerate(markers):
            sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.01)  # 1cm radius
            sphere.vertices += point  # Translate to marker position

            # Create material with blue color
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.0, 0.4, 1.0, 1.0],
                metallicFactor=0.0,
                roughnessFactor=0.5,
            )
            sphere_mesh = pyrender.Mesh.from_trimesh(sphere, material=material)
            scene.add(sphere_mesh)

    # Add skeleton lines if visible
    if show_skeleton:
        for idx1, idx2 in skeleton_connections:
            if idx1 < len(markers) and idx2 < len(markers):
                p1 = markers[idx1]
                p2 = markers[idx2]

                # Create a cylinder between the two points
                # Calculate the vector between points
                vec = p2 - p1
                length = np.linalg.norm(vec)

                if length > 0:
                    # Create a cylinder with small radius
                    cylinder = trimesh.creation.cylinder(
                        radius=0.005,  # Thin line
                        height=length,
                        sections=8,  # Low poly for performance
                    )

                    # Rotate cylinder to align with vector
                    # Default cylinder is along Z-axis
                    z_axis = np.array([0, 0, 1])
                    target_axis = vec / length

                    # Calculate rotation to align z_axis with target_axis
                    if not np.allclose(z_axis, target_axis):
                        # Avoid division by zero
                        if np.allclose(z_axis, -target_axis):
                            # 180 degree rotation around any perpendicular axis
                            rot_axis = np.array([1, 0, 0])
                            rotation = trimesh.transformations.rotation_matrix(np.pi, rot_axis)
                        else:
                            # Calculate rotation using cross product
                            rot_axis = np.cross(z_axis, target_axis)
                            rot_axis = rot_axis / np.linalg.norm(rot_axis)
                            angle = np.arccos(np.clip(np.dot(z_axis, target_axis), -1.0, 1.0))
                            rotation = trimesh.transformations.rotation_matrix(angle, rot_axis)

                        # Apply rotation
                        cylinder.apply_transform(rotation)

                    # Translate to midpoint
                    midpoint = (p1 + p2) / 2
                    translation = trimesh.transformations.translation_matrix(midpoint)
                    cylinder.apply_transform(translation)

                    # Create mesh with gray color
                    material = pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=[0.5, 0.5, 0.5, 1.0],
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    )
                    line_mesh = pyrender.Mesh.from_trimesh(cylinder, material=material)
                    scene.add(line_mesh)

    # Add floor
    floor = create_floor_mesh()
    scene.add(floor)

    # Add directional lights
    light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light1, pose=np.array([[1, 0, 0, 0], [0, 1, 0, 3], [0, 0, 1, 3], [0, 0, 0, 1]]))

    light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
    scene.add(
        light2,
        pose=np.array([[-1, 0, 0, 0], [0, 1, 0, 3], [0, 0, -1, -3], [0, 0, 0, 1]]),
    )

    return scene


def render_frame_with_camera(
    markers,
    skeleton_connections,
    mesh_vertices,
    mesh_faces,
    renderer,
    camera_azimuth=0,
    camera_elevation=15,
    cam_distance=3.0,
    show_markers=True,
    show_skeleton=True,
    show_mesh=True,
):
    """
    Render a single frame with configurable camera angle.

    Args:
        markers: (n_markers, 3) marker positions in SMPL space
        skeleton_connections: List of (idx1, idx2) marker indices
        mesh_vertices: (n_vertices, 3) SMPL mesh vertices
        mesh_faces: (n_faces, 3) SMPL mesh faces
        renderer: pyrender.OffscreenRenderer instance
        camera_azimuth: Camera rotation angle in degrees (0-360, default 0)
        camera_elevation: Camera tilt angle in degrees (-20 to 80, default 15)
        cam_distance: Camera distance from mesh center (default 3.0)
        show_markers: Whether to show marker points
        show_skeleton: Whether to show skeleton lines
        show_mesh: Whether to show SMPL mesh

    Returns:
        RGB numpy array (H, W, 3) uint8
    """
    # Create scene with all elements
    scene = create_scene_with_markers_and_mesh(
        markers,
        skeleton_connections,
        mesh_vertices,
        mesh_faces,
        show_markers,
        show_skeleton,
        show_mesh,
    )

    # Position camera based on azimuth, elevation, and distance
    # Use mesh center as target
    mesh_center = mesh_vertices.mean(axis=0)

    # Convert angles to radians
    azimuth_rad = np.radians(camera_azimuth)
    elevation_rad = np.radians(camera_elevation)

    # Calculate camera position (spherical coordinates)
    cam_x = mesh_center[0] + cam_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    cam_z = mesh_center[2] + cam_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    cam_y = mesh_center[1] + cam_distance * np.sin(elevation_rad)

    # Create camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)

    # Create camera pose looking at the mesh center
    cam_pos = np.array([cam_x, cam_y, cam_z])
    forward = mesh_center - cam_pos
    forward = forward / np.linalg.norm(forward)

    # Use world up vector
    world_up = np.array([0, 1, 0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        # Camera looking straight up or down, use different up vector
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)

    camera_pose = np.eye(4)
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = up
    camera_pose[:3, 2] = -forward
    camera_pose[:3, 3] = cam_pos

    scene.add(camera, pose=camera_pose)

    # Render
    color, _ = renderer.render(scene)

    return color


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GPU-accelerated interactive visualization of markers and SMPL mesh",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_markers_mesh_gpu.py --subject 01 --scene 0
  python visualize_markers_mesh_gpu.py -s 02 -c 1 -m male --gpu-id 1

Controls:
  Frame Slider:   Frame scrubbing (auto-pauses)
  Play/Pause:     Toggle animation playback
  Left/Right:     Rotate camera horizontally (azimuth ±15°)
  Up/Down:        Tilt camera vertically (elevation ±10°)
  Scroll Up:      Zoom in (distance -0.5, min 1.5)
  Scroll Down:    Zoom out (distance +0.5, max 15.0)
  Spacebar:       Play/Pause toggle
  Checkboxes:     Toggle visibility of markers, skeleton, and mesh
        """,
    )

    parser.add_argument(
        "-s",
        "--subject",
        type=str,
        required=True,
        help="Subject ID (e.g., 01, 02, 138)",
    )

    parser.add_argument("-c", "--scene", type=int, required=True, help="Scene number (e.g., 0, 1, 2, 3)")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="neutral",
        choices=["neutral", "male", "female"],
        help="SMPL model type (default: neutral)",
    )

    parser.add_argument("--fps", type=int, default=30, help="Target playback frame rate (default: 30)")

    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Render resolution in pixels (default: 1024)",
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="GPU device ID (0, 1, ...). Prompts if multiple GPUs detected and not specified.",
    )

    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Skip every Nth frame for faster playback (default: 1)",
    )

    parser.add_argument(
        "--test-load",
        action="store_true",
        help="Test data loading only (no GUI)",
    )

    return parser.parse_args()


def main():
    """Main interactive visualization loop."""
    args = parse_arguments()

    print("=" * 60)
    print("GPU-accelerated Markers + SMPL Mesh Visualization")
    print("=" * 60)
    print(f"Subject: {args.subject}")
    print(f"Scene: {args.scene}")
    print(f"Model: {args.model}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print("=" * 60)

    # Load data
    try:
        print("Loading processed markers...")
        marker_data_zup, marker_names, frame_rate = load_processed_markers(args.subject, args.scene)

        print("Loading SMPL parameters...")
        poses, trans, betas, joints, gender = load_smpl_params(args.subject, args.scene)

        # Auto-select model type if detected
        model_type = args.model
        if args.model == "neutral" and gender and gender.lower() in ["male", "female"]:
            print(f"  Auto-selecting {gender} model based on data")
            model_type = gender.lower()

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        data_path = Path("data", "fitted_smpl_all_3")
        print(f"\nAvailable subjects in {data_path}:")
        if data_path.exists():
            subjects = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
            for s in subjects[:15]:
                print(f"  {s}")
            if len(subjects) > 15:
                print(f"  ... and {len(subjects) - 15} more")
        sys.exit(1)

    print(f"\nData loaded:")
    print(f"  Marker data shape: {marker_data_zup.shape}")
    print(f"  Marker names: {len(marker_names)} markers")
    print(f"  SMPL poses shape: {poses.shape}")
    print(f"  Frame rate: {frame_rate} Hz")

    # Check frame count consistency
    n_frames_markers = marker_data_zup.shape[0]
    n_frames_smpl = poses.shape[0]
    if n_frames_markers != n_frames_smpl:
        print(f"Warning: Frame count mismatch: markers {n_frames_markers}, SMPL {n_frames_smpl}")
        n_frames = min(n_frames_markers, n_frames_smpl)
        marker_data_zup = marker_data_zup[:n_frames]
        poses = poses[:n_frames]
        trans = trans[:n_frames]
        print(f"  Using first {n_frames} frames")
    else:
        n_frames = n_frames_markers

    # Transform markers to SMPL space (Z-up → Y-up)
    print("Transforming markers to SMPL space (Y-up)...")
    marker_data_yup = transform_marker_to_smpl_space(marker_data_zup)

    # Pre-compute SMPL vertices
    try:
        smpl_vertices, smpl_faces = precompute_smpl_vertices(poses, trans, betas, model_type)
    except Exception as e:
        print(f"Error computing SMPL vertices: {e}")
        print("Falling back to using SMPL joints only...")
        # Use joints as proxy for mesh
        smpl_vertices = joints
        # Create dummy faces (won't be used for wireframe)
        smpl_faces = np.array([[0, 1, 2]])

    # Validate skeleton connections
    print("\nValidating skeleton connections...")
    skeleton_connections, missing_markers = validate_skeleton_connections(marker_names, MARKER_SKELETON)

    if missing_markers:
        print(f"  Warning: Missing markers: {missing_markers}")
    print(f"  Valid connections: {len(skeleton_connections)}")

    if not skeleton_connections:
        print("Error: No valid skeleton connections found!")
        sys.exit(1)

    # Downsample frames if requested
    marker_data_yup = marker_data_yup[:: args.downsample]
    smpl_vertices = smpl_vertices[:: args.downsample]
    total_frames = len(marker_data_yup)

    print(f"\nTotal frames: {total_frames}")
    if args.downsample > 1:
        print(f"(downsampled by {args.downsample}x)")

    # Setup GPU rendering backend FIRST
    print("Setting up GPU rendering backend...")
    setup_rendering_backend(args.gpu_id)
    renderer = setup_renderer_live(args.resolution, args.resolution)

    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.20)  # More space for controls
    ax.axis("off")
    ax.set_title("Markers + SMPL Mesh Viewer (Arrow keys: rotate, Scroll: zoom, Space: pause)")

    # UI State
    ui_state = {
        "frame": 0,
        "paused": False,
        "anim": None,
        "camera_azimuth": 0,
        "camera_elevation": 15,
        "cam_distance": 3.0,
        "show_markers": True,
        "show_skeleton": True,
        "show_mesh": True,
        "renderer": renderer,
    }

    # Pre-render first frame
    print("Rendering first frame...")
    initial_frame = render_frame_with_camera(
        marker_data_yup[0],
        skeleton_connections,
        smpl_vertices[0],
        smpl_faces,
        renderer,
        ui_state["camera_azimuth"],
        ui_state["camera_elevation"],
        ui_state["cam_distance"],
        ui_state["show_markers"],
        ui_state["show_skeleton"],
        ui_state["show_mesh"],
    )
    img_display = ax.imshow(initial_frame)

    # Update visuals function
    def update_visuals(frame_idx):
        """Render and display a specific frame."""
        frame_rgb = render_frame_with_camera(
            marker_data_yup[frame_idx],
            skeleton_connections,
            smpl_vertices[frame_idx],
            smpl_faces,
            renderer,
            ui_state["camera_azimuth"],
            ui_state["camera_elevation"],
            ui_state["cam_distance"],
            ui_state["show_markers"],
            ui_state["show_skeleton"],
            ui_state["show_mesh"],
        )
        img_display.set_data(frame_rgb)
        ax.set_title(
            f"Frame {frame_idx * args.downsample} / {total_frames * args.downsample - 1} | "
            f"Az: {ui_state['camera_azimuth']}° El: {ui_state['camera_elevation']}° "
            f"Dist: {ui_state['cam_distance']:.1f}"
        )

    # Animation frame callback
    def on_frame(frame):
        if ui_state["paused"]:
            return [img_display]

        ui_state["frame"] = frame
        frame_slider.eventson = False
        frame_slider.set_val(frame)
        frame_slider.eventson = True

        update_visuals(frame)
        return [img_display]

    # Keyboard handler for camera control
    def on_key(event):
        if event.key == "left":
            ui_state["camera_azimuth"] = (ui_state["camera_azimuth"] - 15) % 360
        elif event.key == "right":
            ui_state["camera_azimuth"] = (ui_state["camera_azimuth"] + 15) % 360
        elif event.key == "up":
            ui_state["camera_elevation"] = min(ui_state["camera_elevation"] + 10, 80)
        elif event.key == "down":
            ui_state["camera_elevation"] = max(ui_state["camera_elevation"] - 10, -20)
        elif event.key == " ":
            toggle_pause(None)
            return
        else:
            return

        # Re-render current frame with new camera angle
        update_visuals(ui_state["frame"])
        fig.canvas.draw_idle()

    # Scroll handler for zoom
    def on_scroll(event):
        zoom_factor = 0.5
        if event.button == "up":  # Scroll up -> zoom in
            ui_state["cam_distance"] = max(1.5, ui_state["cam_distance"] - zoom_factor)
        elif event.button == "down":  # Scroll down -> zoom out
            ui_state["cam_distance"] = min(15.0, ui_state["cam_distance"] + zoom_factor)

        # Re-render current frame with new zoom
        update_visuals(ui_state["frame"])
        fig.canvas.draw_idle()

    # Connect event handlers
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    # Frame slider widget
    ax_frame_slider = plt.axes((0.2, 0.10, 0.55, 0.03))
    frame_slider = Slider(ax_frame_slider, "Frame", 0, total_frames - 1, valinit=0, valfmt="%d")

    def update_frame_slider(val):
        frame = int(frame_slider.val)
        ui_state["frame"] = frame
        ui_state["paused"] = True
        button.label.set_text("Play")
        if ui_state["anim"]:
            ui_state["anim"].event_source.stop()

        update_visuals(frame)
        fig.canvas.draw_idle()

    frame_slider.on_changed(update_frame_slider)

    # Visibility checkboxes
    ax_checkbox = plt.axes((0.2, 0.04, 0.2, 0.05))
    checkboxes = CheckButtons(ax_checkbox, ["Markers", "Skeleton", "Mesh"], [True, True, True])

    def update_checkbox(label):
        if label == "Markers":
            ui_state["show_markers"] = not ui_state["show_markers"]
            print(f"Markers visibility: {ui_state['show_markers']}")
        elif label == "Skeleton":
            ui_state["show_skeleton"] = not ui_state["show_skeleton"]
            print(f"Skeleton visibility: {ui_state['show_skeleton']}")
        elif label == "Mesh":
            ui_state["show_mesh"] = not ui_state["show_mesh"]
            print(f"Mesh visibility: {ui_state['show_mesh']}")

        # Re-render with updated visibility
        update_visuals(ui_state["frame"])
        fig.canvas.draw_idle()

    checkboxes.on_clicked(update_checkbox)

    # Play/Pause button
    ax_button = plt.axes((0.8, 0.10, 0.1, 0.04))
    button = Button(ax_button, "Pause")

    def toggle_pause(event):
        ui_state["paused"] = not ui_state["paused"]
        if ui_state["paused"]:
            button.label.set_text("Play")
            if ui_state["anim"]:
                ui_state["anim"].event_source.stop()
        else:
            button.label.set_text("Pause")
            if ui_state["anim"]:
                ui_state["anim"].event_source.start()

    button.on_clicked(toggle_pause)

    # Create animation
    interval = max(1000 // args.fps, 30)  # Minimum 30ms between frames
    print(f"\nStarting animation at {1000 // interval:.0f} FPS (interval: {interval}ms)")
    print("\nControls:")
    print("  Left/Right arrows: Rotate camera horizontally (azimuth ±15°)")
    print("  Up/Down arrows:    Tilt camera vertically (elevation ±10°)")
    print("  Scroll Up/Down:    Zoom in/out (distance ±0.5)")
    print("  Spacebar:          Play/Pause toggle")
    print("  Frame Slider:      Seek to specific frame")
    print("  Checkboxes:        Toggle visibility of markers, skeleton, and mesh")
    print("  Note: Visibility toggles require scene recreation - will be implemented")

    anim = FuncAnimation(
        fig,
        on_frame,
        frames=range(total_frames),
        interval=interval,
        blit=False,
        repeat=True,
    )
    ui_state["anim"] = anim

    # Cleanup on close
    def on_close(event):
        print("\nCleaning up renderer...")
        renderer.delete()

    fig.canvas.mpl_connect("close_event", on_close)

    plt.show()


if __name__ == "__main__":
    # Simple test to verify data loading without opening GUI
    import sys

    args = parse_arguments()

    if args.test_load:
        print("Testing data loading only...")

        try:
            print("Loading processed markers...")
            marker_data_zup, marker_names, frame_rate = load_processed_markers(args.subject, args.scene)

            print("Loading SMPL parameters...")
            poses, trans, betas, joints, gender = load_smpl_params(args.subject, args.scene)

            print(f"Data loaded successfully!")
            print(f"  Marker data shape: {marker_data_zup.shape}")
            print(f"  Marker names: {len(marker_names)} markers")
            print(f"  SMPL poses shape: {poses.shape}")
            print(f"  Frame rate: {frame_rate} Hz")

            # Test coordinate transformation
            print("\nTesting coordinate transformation...")
            marker_data_yup = transform_marker_to_smpl_space(marker_data_zup)
            print(f"  Original Z-up shape: {marker_data_zup.shape}")
            print(f"  Transformed Y-up shape: {marker_data_yup.shape}")
            print(f"  Sample marker (frame 0, marker 0):")
            print(f"    Z-up: {marker_data_zup[0, 0]}")
            print(f"    Y-up: {marker_data_yup[0, 0]}")

            # Test skeleton validation
            print("\nTesting skeleton validation...")
            skeleton_connections, missing_markers = validate_skeleton_connections(marker_names, MARKER_SKELETON)
            print(f"  Valid connections: {len(skeleton_connections)}")
            if missing_markers:
                print(f"  Missing markers: {missing_markers}")

            print("\nAll tests passed! Data loading and transformation working correctly.")

        except Exception as e:
            print(f"Error during test: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
    else:
        main()
