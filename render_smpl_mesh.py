#!/usr/bin/env python3
"""
SMPL Mesh Visualization Script
Renders SMPL body model with motion capture data to MP4 video.

Usage:
    # Basic usage - auto-detects GPU
    python3 render_smpl_mesh.py --subject 01 --scene 1

    # Specify GPU manually (useful when multiple GPUs detected)
    python3 render_smpl_mesh.py --subject 02 --scene 0 --gpu-id 1

    # Full example with all options
    python3 render_smpl_mesh.py -s 138 -c 2 -m female --gpu-id 0 --fps 60
"""

import os
import sys

import argparse
import numpy as np
import torch
import trimesh
import pyrender
import cv2
from tqdm import tqdm
import subprocess
import re
from pathlib import Path

# smplx is imported inside initialize_smpl_model() to ensure correct working directory


def detect_gpus():
    """
    Detect available GPUs and return their information.

    Returns:
        list: List of dicts with 'id', 'device', 'vendor', 'model' keys
    """
    gpus = []
    dri_path = Path('/dev/dri')

    if not dri_path.exists():
        return gpus

    # Find all card devices
    cards = sorted([c for c in dri_path.glob('card*') if c.name.startswith('card') and c.name[4:].isdigit()])

    for card_idx, card in enumerate(cards):
        gpu_info = {
            'id': card_idx,
            'device': str(card),
            'vendor': 'Unknown',
            'model': 'Unknown'
        }

        try:
            # Get device information using udevadm
            result = subprocess.run(
                ['udevadm', 'info', '--query=all', f'--name={card}'],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                output = result.stdout

                # Extract PCI path
                pci_match = re.search(r'ID_PATH=pci-([\S]+)', output)
                if pci_match:
                    pci_path = pci_match.group(1)

                    # Get GPU info from lspci
                    lspci_result = subprocess.run(
                        ['lspci', '-v', '-s', pci_path],
                        capture_output=True,
                        text=True,
                        timeout=2
                    )

                    if lspci_result.returncode == 0:
                        lspci_output = lspci_result.stdout

                        # Parse vendor and model from VGA controller line
                        vga_match = re.search(r'VGA compatible controller: (.+)', lspci_output)
                        if vga_match:
                            full_name = vga_match.group(1).strip()

                            # Split vendor and model
                            if 'NVIDIA' in full_name:
                                gpu_info['vendor'] = 'NVIDIA'
                                gpu_info['model'] = full_name.replace('NVIDIA Corporation', '').strip()
                            elif 'AMD' in full_name or 'ATI' in full_name:
                                gpu_info['vendor'] = 'AMD'
                                # Clean up AMD naming
                                model = re.sub(r'Advanced Micro Devices, Inc\.\s*\[AMD/ATI\]\s*', '', full_name)
                                model = re.sub(r'\s*\(prog-if.*?\).*', '', model)
                                model = re.sub(r'\s*\(rev.*?\).*', '', model)
                                gpu_info['model'] = model.strip()
                            elif 'Intel' in full_name:
                                gpu_info['vendor'] = 'Intel'
                                gpu_info['model'] = full_name.replace('Intel Corporation', '').strip()
                            else:
                                gpu_info['model'] = full_name

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
                ['glxinfo'],
                capture_output=True,
                text=True,
                timeout=2,
                env={**os.environ, 'DRI_PRIME': str(gpu['id'])}
            )
            if result.returncode == 0 and gpu['vendor'] in result.stdout:
                status = " [drivers OK]"
        except Exception:
            pass

        print(f"  [{gpu['id']}] {gpu['vendor']} - {gpu['model']}{status}")
        print(f"      Device: {gpu['device']}")

    print("=" * 60)

    while True:
        try:
            choice = input(f"Select GPU [0-{len(gpus)-1}]: ").strip()
            gpu_id = int(choice)
            if 0 <= gpu_id < len(gpus):
                selected = gpus[gpu_id]
                print(f"\nSelected: {selected['vendor']} - {selected['model']}")
                return gpu_id
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(gpus)-1}")
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
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    # Detect available GPUs
    gpus = detect_gpus()

    if len(gpus) == 0:
        print("Warning: No GPU devices detected. Trying default EGL device...")
        return None

    if len(gpus) == 1:
        # Only one GPU, use it automatically
        gpu_id = 0
        print(f"Using GPU: {gpus[0]['vendor']} - {gpus[0]['model']}")
        os.environ['EGL_DEVICE_ID'] = str(gpu_id)
        return gpu_id

    # Multiple GPUs detected
    if gpu_id is None:
        # Prompt user to select
        gpu_id = select_gpu(gpus)
    elif gpu_id >= len(gpus):
        print(f"Error: GPU ID {gpu_id} not found. Available GPUs: 0-{len(gpus)-1}")
        sys.exit(1)
    else:
        # Use specified GPU
        print(f"Using GPU: {gpus[gpu_id]['vendor']} - {gpus[gpu_id]['model']}")

    os.environ['EGL_DEVICE_ID'] = str(gpu_id)
    return gpu_id


def setup_renderer(width=1024, height=1024):
    """
    Initialize offscreen renderer.

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


def load_data(subject, scene):
    """
    Load SMPL shape and motion data.

    Args:
        subject: Subject ID (e.g., '01', '02', '138')
        scene: Scene number (e.g., 0, 1, 2, 3)

    Returns:
        betas, poses, trans: Shape and motion parameters
    """
    print(f"Loading data for Subject {subject}, Scene {scene}...")

    # Format subject ID with zero padding if needed
    subj_id = f"SUBJ{subject}"

    # Load shape parameters (betas) - constant for subject
    # Data is in parent directory
    data_dir = os.path.join('data', 'fitted_smpl_all_3', subj_id)
    betas_path = os.path.join(data_dir, 'betas.npy')

    if not os.path.exists(betas_path):
        raise FileNotFoundError(f"Shape file not found: {betas_path}")

    betas = np.load(betas_path)
    betas = torch.tensor(betas, dtype=torch.float32)

    # Ensure betas is shape (1, 10)
    if betas.ndim == 1:
        betas = betas.unsqueeze(0)
    if betas.shape[1] < 10:
        # Pad with zeros if needed
        padding = torch.zeros(1, 10 - betas.shape[1])
        betas = torch.cat([betas, padding], dim=1)

    print(f"  Betas shape: {betas.shape}")

    # Load motion data (poses and translations)
    # Format: SUBJ1_0, SUBJ2_1, etc.
    subj_num = subject.lstrip('0') or '0'  # Remove leading zeros, '01' -> '1'
    motion_path = os.path.join(data_dir, f'SUBJ{subj_num}_{scene}_smpl_params.npz')

    if not os.path.exists(motion_path):
        raise FileNotFoundError(f"Motion file not found: {motion_path}")

    motion_data = np.load(motion_path)

    print("  Motion data keys:", list(motion_data.keys()))

    # Extract pose and translation data
    # Handle different possible key names
    if 'poses' in motion_data:
        poses = motion_data['poses']
    elif 'body_pose' in motion_data and 'global_orient' in motion_data:
        # Concatenate global orientation and body pose
        global_orient = motion_data['global_orient']
        body_pose = motion_data['body_pose']
        poses = np.concatenate([global_orient, body_pose], axis=-1)
    else:
        raise ValueError(f"Cannot find pose data in keys: {list(motion_data.keys())}")

    if 'trans' in motion_data:
        trans = motion_data['trans']
    elif 'transl' in motion_data:
        trans = motion_data['transl']
    else:
        raise ValueError(f"Cannot find translation data in keys: {list(motion_data.keys())}")

    poses = torch.tensor(poses, dtype=torch.float32)
    trans = torch.tensor(trans, dtype=torch.float32)

    print(f"  Poses shape: {poses.shape}")
    print(f"  Trans shape: {trans.shape}")
    print(f"  Total frames: {len(poses)}")

    # Extract gender from motion data if available
    gender = None
    if 'gender' in motion_data:
        gender = str(motion_data['gender'])
        print(f"  Detected gender: {gender}")

    return betas, poses, trans, gender


def initialize_smpl_model(model_type='neutral'):
    """
    Initialize SMPL body model using smplx library.

    Args:
        model_type: Model gender type ('neutral', 'male', or 'female')

    Returns:
        smplx.SMPL model instance
    """
    print(f"Initializing SMPL model (type: {model_type})...")

    if model_type.lower() not in ['neutral', 'male', 'female']:
        raise ValueError(f"Invalid model type: {model_type}. Must be 'neutral', 'male', or 'female'")

    # Load model using config from visualize_joints (already in that directory)
    import smplx
    model = smplx.create(
        model_path=os.path.join('data', 'smpl'),
        model_type='smpl',
        gender=model_type.lower(),
        ext='pkl',
        batch_size=1
    )

    print("  SMPL model loaded successfully")
    return model


def create_floor_mesh():
    """Create a grid floor for visual reference."""
    # Create a simple floor plane
    grid_size = 4.0
    grid_step = 0.5

    vertices = []
    faces = []

    # Create grid vertices
    num_lines = int(grid_size / grid_step) * 2 + 1
    for i in range(num_lines):
        x = -grid_size + i * grid_step
        vertices.append([x, 0, -grid_size])
        vertices.append([x, 0, grid_size])

    for i in range(num_lines):
        z = -grid_size + i * grid_step
        vertices.append([-grid_size, 0, z])
        vertices.append([grid_size, 0, z])

    vertices = np.array(vertices)

    # Create a simple floor plane mesh
    floor_vertices = np.array([
        [-grid_size, 0, -grid_size],
        [grid_size, 0, -grid_size],
        [grid_size, 0, grid_size],
        [-grid_size, 0, grid_size]
    ])

    floor_faces = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ])

    floor_mesh = trimesh.Trimesh(vertices=floor_vertices, faces=floor_faces)
    floor_mesh.visual.vertex_colors = [200, 200, 200, 100]  # Light gray, semi-transparent

    return pyrender.Mesh.from_trimesh(floor_mesh, smooth=False)


def render_frame(smpl_model, betas, pose, transl, renderer):
    """Render a single frame with the SMPL mesh."""

    # Handle pose dimensions
    if pose.ndim == 1:
        pose = pose.unsqueeze(0)

    if transl.ndim == 1:
        transl = transl.unsqueeze(0)

    # Split pose into global orientation and body pose
    # SMPL expects: global_orient (3,) and body_pose (69,)
    global_orient = pose[:, :3]
    body_pose = pose[:, 3:]

    # Forward pass through SMPL
    with torch.no_grad():
        output = smpl_model(
            betas=betas,
            global_orient=global_orient,
            body_pose=body_pose,
            transl=transl,
            return_verts=True
        )

    vertices = output.vertices.detach().cpu().numpy()[0]
    faces = smpl_model.faces

    # Create trimesh object
    body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    body_mesh.visual.vertex_colors = [174, 199, 232, 255]  # Light blue color

    # Create pyrender mesh
    mesh = pyrender.Mesh.from_trimesh(body_mesh, smooth=True)

    # Create scene
    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[255, 255, 255])

    # Add body mesh
    scene.add(mesh)

    # Add floor
    floor = create_floor_mesh()
    scene.add(floor)

    # Add directional light from multiple angles
    light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light1, pose=np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 3],
        [0, 0, 1, 3],
        [0, 0, 0, 1]
    ]))

    light2 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.5)
    scene.add(light2, pose=np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 3],
        [0, 0, -1, -3],
        [0, 0, 0, 1]
    ]))

    # Position camera to follow the person
    # Get the center of the mesh for camera targeting
    mesh_center = vertices.mean(axis=0)

    # Camera positioned at an angle to see the person from the side
    cam_distance = 5.0
    cam_height = 1.5
    cam_x = mesh_center[0] + cam_distance * 0.7  # Slightly forward
    cam_y = cam_height
    cam_z = mesh_center[2] + cam_distance  # To the side

    # Look at the mesh center
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)

    # Create camera pose looking at the mesh
    forward = mesh_center - np.array([cam_x, cam_y, cam_z])
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, np.array([0, 1, 0]))
    right = right / np.linalg.norm(right)

    up = np.cross(right, forward)

    camera_pose = np.eye(4)
    camera_pose[:3, 0] = right
    camera_pose[:3, 1] = up
    camera_pose[:3, 2] = -forward
    camera_pose[:3, 3] = [cam_x, cam_y, cam_z]

    scene.add(camera, pose=camera_pose)

    # Render
    color, _ = renderer.render(scene)

    return color


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Render SMPL mesh visualization from motion capture data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render Subject 01, Scene 1 with neutral model
  python3 render_smpl_mesh.py --subject 01 --scene 1

  # Render Subject 02, Scene 0 with male model
  python3 render_smpl_mesh.py --subject 02 --scene 0 --model male

  # Short form
  python3 render_smpl_mesh.py -s 138 -c 2 -m female
        """
    )

    parser.add_argument(
        '-s', '--subject',
        type=str,
        required=True,
        help='Subject ID (e.g., 01, 02, 138)'
    )

    parser.add_argument(
        '-c', '--scene',
        type=int,
        required=True,
        help='Scene number (e.g., 0, 1, 2, 3)'
    )

    parser.add_argument(
        '-m', '--model',
        type=str,
        default='neutral',
        choices=['neutral', 'male', 'female'],
        help='SMPL model type (default: neutral)'
    )

    parser.add_argument(
        '--max-frames',
        type=int,
        default=3000,
        help='Maximum number of frames to render (default: 300, 0 for all)'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Output video frame rate (default: 30)'
    )

    parser.add_argument(
        '--resolution',
        type=int,
        default=1024,
        help='Output video resolution (square, default: 1024)'
    )

    parser.add_argument(
        '--gpu-id',
        type=int,
        default=None,
        help='GPU device ID to use (prompts if multiple GPUs detected and not specified)'
    )

    return parser.parse_args()


def main():
    """Main rendering pipeline."""
    # Parse command line arguments
    args = parse_arguments()

    # Setup rendering backend first
    print("Setting up rendering backend...")
    setup_rendering_backend(args.gpu_id)

    print("=" * 60)
    print("SMPL Mesh Visualization Script")
    print("=" * 60)
    print(f"Subject: {args.subject}")
    print(f"Scene: {args.scene}")
    print(f"Model: {args.model}")
    print("=" * 60)

    # Create output directory if it doesn't exist
    os.makedirs('outputs/videos', exist_ok=True)

    # Load data
    try:
        betas, poses, trans, detected_gender = load_data(args.subject, args.scene)

        # If model type is not specified and we detected gender, use it
        model_type = args.model
        if args.model == 'neutral' and detected_gender and detected_gender.lower() in ['male', 'female']:
            print(f"  Auto-selecting {detected_gender} model based on data")
            model_type = detected_gender.lower()

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print(f"\nAvailable subjects can be found in: {os.path.join('data', 'fitted_smpl_all_3')}")
        sys.exit(1)

    # Initialize SMPL model
    try:
        smpl_model = initialize_smpl_model(model_type)
    except (ValueError, FileNotFoundError) as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Setup renderer
    print("Setting up renderer...")
    renderer = setup_renderer(args.resolution, args.resolution)

    # Determine number of frames to render
    if args.max_frames == 0:
        num_frames = len(poses)
    else:
        num_frames = min(args.max_frames, len(poses))
    print(f"\nRendering {num_frames} frames...")

    # Setup video writer with descriptive filename
    output_filename = f'SUBJ{args.subject}_scene{args.scene}_{model_type}_visualization.mp4'
    output_path = f'outputs/videos/{output_filename}'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, args.fps, (args.resolution, args.resolution))

    # Render loop
    frames = []
    for frame_idx in tqdm(range(num_frames), desc="Rendering"):
        # Get current pose and translation
        current_pose = poses[frame_idx]
        current_trans = trans[frame_idx]

        # Render frame
        rgb_frame = render_frame(smpl_model, betas, current_pose, current_trans, renderer)

        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Write frame
        video_writer.write(bgr_frame)

    # Release video writer
    video_writer.release()
    renderer.delete()

    print(f"\n{'=' * 60}")
    print(f"Video saved successfully to {output_path}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
