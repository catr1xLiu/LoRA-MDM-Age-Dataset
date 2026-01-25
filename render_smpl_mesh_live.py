#!/usr/bin/env python3
"""
SMPL Mesh Live Visualization Script
Interactive rendering of SMPL body model with motion capture data.

Displays SMPL mesh in a live matplotlib window with playback controls,
combining the rendering from render_smpl_mesh.py with the interactive
UI pattern from 6_npz_motion_to_gif.py.

Usage:
    # Basic usage (will prompt for GPU selection if multiple detected)
    python3 render_smpl_mesh_live.py --subject 01 --scene 2

    # Specify GPU manually
    python3 render_smpl_mesh_live.py -s 01 -c 2 --gpu-id 1

Controls:
    Frame Slider:   Frame scrubbing (auto-pauses)
    Beta Slider:    Adjust body shape (uniform betas, -10 to +10, affects "fatness")
    Override:       Checkbox to enable/disable beta slider override (unchecked = original betas)
    Play/Pause:     Toggle animation playback
    Left/Right:     Rotate camera horizontally (azimuth ±15°)
    Up/Down:        Tilt camera vertically (elevation ±10°, range: -20° to 80°)
    Scroll Up:      Zoom in (decrease distance by 0.5, min: 1.5)
    Scroll Down:    Zoom out (increase distance by 0.5, max: 15.0)
    Spacebar:       Play/Pause toggle
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
from trimesh.visual import ColorVisuals
import pyrender
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.animation import FuncAnimation


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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Interactive SMPL mesh visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 render_smpl_mesh_live.py --subject 01 --scene 2
  python3 render_smpl_mesh_live.py -s 02 -c 0 -m male --gpu-id 1

Controls:
  Frame Slider:   Frame scrubbing (auto-pauses)
  Beta Slider:    Adjust body shape (-10 to +10, uniform "fatness")
  Play/Pause:     Toggle animation playback
  Left/Right:     Rotate camera horizontally (azimuth ±15°)
  Up/Down:        Tilt camera vertically (elevation ±10°)
  Scroll Up:      Zoom in (distance -0.5, min 1.5)
  Scroll Down:    Zoom out (distance +0.5, max 15.0)
  Spacebar:       Play/Pause toggle
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
        '--fps',
        type=int,
        default=30,
        help='Target playback frame rate (default: 30)'
    )

    parser.add_argument(
        '--resolution',
        type=int,
        default=1024,
        help='Render resolution in pixels (default: 1024)'
    )

    parser.add_argument(
        '--gpu-id',
        type=int,
        default=None,
        help='GPU device ID (0, 1, ...). Prompts if multiple GPUs detected and not specified.'
    )

    parser.add_argument(
        '--downsample',
        type=int,
        default=1,
        help='Skip every Nth frame for faster playback (default: 1)'
    )

    return parser.parse_args()


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


def load_data(subject, scene):
    """
    Load SMPL shape and motion data.

    Args:
        subject: Subject ID (e.g., '01', '02', '138')
        scene: Scene number (e.g., 0, 1, 2, 3)

    Returns:
        betas, poses, trans, gender: Shape and motion parameters
    """
    print(f"Loading data for Subject {subject}, Scene {scene}...")

    # Format subject ID with zero padding if needed
    subj_id = f"SUBJ{subject}"

    # Data directory relative to script location (in parent directory)
    data_dir = os.path.join('data', 'fitted_smpl_all_3', subj_id)

    # Load shape parameters (betas) - constant for subject
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
    if 'poses' in motion_data:
        poses = motion_data['poses']
    elif 'body_pose' in motion_data and 'global_orient' in motion_data:
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

    import smplx
    model = smplx.create(
        "data/smpl/",
        model_type='smpl',
        gender=model_type.lower(),
        ext='pkl',
        batch_size=1
    )

    print("  SMPL model loaded successfully")
    return model


def create_floor_mesh():
    """Create a simple flat floor plane for visual reference."""
    grid_size = 4.0

    # Create a simple quad at y=0
    vertices = np.array([
        [-grid_size, 0, -grid_size],
        [grid_size, 0, -grid_size],
        [-grid_size, 0, grid_size],
        [grid_size, 0, grid_size]
    ])

    # Two triangles with CCW winding for upward normals
    faces = np.array([
        [0, 2, 1],  # First triangle
        [1, 2, 3]   # Second triangle
    ])

    floor_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    floor_mesh.visual.vertex_colors = [150, 150, 180, 150]  # Light blue-gray

    # Material with double-sided rendering
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=0.8,
        alphaMode='BLEND',
        doubleSided=True
    )

    return pyrender.Mesh.from_trimesh(floor_mesh, material=material, smooth=False)


def render_frame_with_camera(smpl_model, betas, pose, transl, renderer,
                              camera_azimuth=0, camera_elevation=15, cam_distance=3.0):
    """
    Render a single frame with configurable camera angle.

    Args:
        smpl_model: SimpleSMPL model instance
        betas: Shape parameters (1, 10)
        pose: Pose parameters for this frame (72,)
        transl: Translation for this frame (3,)
        renderer: pyrender.OffscreenRenderer instance
        camera_azimuth: Camera rotation angle in degrees (0-360, default 0)
        camera_elevation: Camera tilt angle in degrees (-20 to 80, default 15)
        cam_distance: Camera distance from mesh center (default 3.0)

    Returns:
        RGB numpy array (H, W, 3) uint8
    """
    # Handle pose dimensions
    if pose.ndim == 1:
        pose = pose.unsqueeze(0)
    if transl.ndim == 1:
        transl = transl.unsqueeze(0)

    # Split pose into global orientation and body pose
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
    body_mesh.visual.vertex_colors = [174, 199, 232, 255]  # type: ignore # Light blue color

    # Create pyrender mesh
    mesh = pyrender.Mesh.from_trimesh(body_mesh, smooth=True)

    # Create scene
    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[255, 255, 255])

    # Add body mesh
    scene.add(mesh)

    # Add floor
    floor = create_floor_mesh()
    scene.add(floor)

    # Add directional lights
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

    # Position camera based on azimuth, elevation, and distance
    mesh_center = vertices.mean(axis=0)

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


def main(args):
    """Main interactive visualization loop."""
    print("=" * 60)
    print("SMPL Mesh Live Visualization")
    print("=" * 60)
    print(f"Subject: {args.subject}")
    print(f"Scene: {args.scene}")
    print(f"Model: {args.model}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print("=" * 60)

    # Load data
    try:
        betas, poses, trans, detected_gender = load_data(args.subject, args.scene)

        # Auto-select model type if detected
        model_type = args.model
        if args.model == 'neutral' and detected_gender and detected_gender.lower() in ['male', 'female']:
            print(f"  Auto-selecting {detected_gender} model based on data")
            model_type = detected_gender.lower()

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        data_path = Path('data', 'fitted_smpl_all_3')
        print(f"\nAvailable subjects in {data_path}:")
        if data_path.exists():
            subjects = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
            for s in subjects[:15]:
                print(f"  {s}")
            if len(subjects) > 15:
                print(f"  ... and {len(subjects) - 15} more")
        sys.exit(1)

    # Initialize SMPL model
    try:
        smpl_model = initialize_smpl_model(model_type)
    except (ValueError, FileNotFoundError) as e:
        print(f"\nError: {e}")
        sys.exit(1)

    # Setup renderer
    print("Setting up renderer...")
    renderer = setup_renderer_live(args.resolution, args.resolution)

    # Downsample frames if requested
    poses = poses[::args.downsample]
    trans = trans[::args.downsample]
    total_frames = len(poses)

    print(f"\nTotal frames: {total_frames}")
    if args.downsample > 1:
        print(f"(downsampled by {args.downsample}x)")

    # Setup matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.20)  # More space for two sliders
    ax.axis('off')
    ax.set_title('SMPL Mesh Viewer (Arrow keys: rotate, Scroll: zoom, Space: pause)')

    # UI State
    ui_state = {
        'frame': 0,
        'paused': False,
        'anim': None,
        'camera_azimuth': 0,
        'camera_elevation': 15,
        'cam_distance': 3.0,  # Closer default for better visibility
        'beta_value': 0.0,  # Uniform beta value for all 10 shape params
        'original_betas': betas.clone(),  # Original betas from npz data
        'betas': betas.clone(),  # Mutable copy of betas (used for rendering)
        'override_betas': False  # Whether to use slider value or original betas
    }

    # Pre-render first frame
    print("Rendering first frame...")
    initial_frame = render_frame_with_camera(
        smpl_model, ui_state['betas'], poses[0], trans[0], renderer,
        ui_state['camera_azimuth'],
        ui_state['camera_elevation'],
        ui_state['cam_distance']
    )
    img_display = ax.imshow(initial_frame)

    # Update visuals function
    def update_visuals(frame_idx):
        """Render and display a specific frame."""
        frame_rgb = render_frame_with_camera(
            smpl_model, ui_state['betas'], poses[frame_idx], trans[frame_idx], renderer,
            ui_state['camera_azimuth'],
            ui_state['camera_elevation'],
            ui_state['cam_distance']
        )
        img_display.set_data(frame_rgb)
        beta_status = f'Beta: {ui_state["beta_value"]:.1f}' if ui_state['override_betas'] else 'Beta: original'
        ax.set_title(f'Frame {frame_idx * args.downsample} / {total_frames * args.downsample - 1} | '
                     f'Az: {ui_state["camera_azimuth"]}° El: {ui_state["camera_elevation"]}° '
                     f'Dist: {ui_state["cam_distance"]:.1f} | {beta_status}')

    # Animation frame callback
    def on_frame(frame):
        if ui_state['paused']:
            return [img_display]

        ui_state['frame'] = frame
        frame_slider.eventson = False
        frame_slider.set_val(frame)
        frame_slider.eventson = True

        update_visuals(frame)
        return [img_display]

    # Keyboard handler for camera control
    def on_key(event):
        if event.key == 'left':
            ui_state['camera_azimuth'] = (ui_state['camera_azimuth'] - 15) % 360
        elif event.key == 'right':
            ui_state['camera_azimuth'] = (ui_state['camera_azimuth'] + 15) % 360
        elif event.key == 'up':
            ui_state['camera_elevation'] = min(ui_state['camera_elevation'] + 10, 80)
        elif event.key == 'down':
            ui_state['camera_elevation'] = max(ui_state['camera_elevation'] - 10, -20)
        elif event.key == ' ':
            toggle_pause(None)
            return
        else:
            return

        # Re-render current frame with new camera angle
        update_visuals(ui_state['frame'])
        fig.canvas.draw_idle()

    # Scroll handler for zoom
    def on_scroll(event):
        zoom_factor = 0.5
        if event.button == 'up':  # Scroll up -> zoom in
            ui_state['cam_distance'] = max(1.5, ui_state['cam_distance'] - zoom_factor)
        elif event.button == 'down':  # Scroll down -> zoom out
            ui_state['cam_distance'] = min(15.0, ui_state['cam_distance'] + zoom_factor)

        # Re-render current frame with new zoom
        update_visuals(ui_state['frame'])
        fig.canvas.draw_idle()

    # Connect event handlers
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # Frame slider widget
    ax_frame_slider = plt.axes((0.2, 0.10, 0.55, 0.03))
    frame_slider = Slider(ax_frame_slider, 'Frame', 0, total_frames - 1, valinit=0, valfmt='%d')

    def update_frame_slider(val):
        frame = int(frame_slider.val)
        ui_state['frame'] = frame
        ui_state['paused'] = True
        button.label.set_text("Play")
        if ui_state['anim']:
            ui_state['anim'].event_source.stop()

        update_visuals(frame)
        fig.canvas.draw_idle()

    frame_slider.on_changed(update_frame_slider)

    # Beta slider widget (uniform body shape - "fatness")
    ax_beta_slider = plt.axes((0.2, 0.05, 0.55, 0.03))
    beta_slider = Slider(ax_beta_slider, 'Beta', -10.0, 10.0, valinit=0.0, valfmt='%.1f')

    def update_beta_slider(val):
        beta_val = beta_slider.val
        ui_state['beta_value'] = beta_val
        # Only apply slider value if override is enabled
        if ui_state['override_betas']:
            ui_state['betas'] = torch.full((1, 10), beta_val, dtype=torch.float32)
            update_visuals(ui_state['frame'])
            fig.canvas.draw_idle()

    beta_slider.on_changed(update_beta_slider)

    # Beta override checkbox
    ax_checkbox = plt.axes((0.8, 0.04, 0.15, 0.05))
    checkbox = CheckButtons(ax_checkbox, ['Override'], [False])

    def update_checkbox(_label):
        ui_state['override_betas'] = not ui_state['override_betas']
        if ui_state['override_betas']:
            # Apply slider value
            ui_state['betas'] = torch.full((1, 10), ui_state['beta_value'], dtype=torch.float32)
        else:
            # Restore original betas
            ui_state['betas'] = ui_state['original_betas'].clone()

        update_visuals(ui_state['frame'])
        fig.canvas.draw_idle()

    checkbox.on_clicked(update_checkbox)

    # Play/Pause button
    ax_button = plt.axes((0.8, 0.10, 0.1, 0.04))
    button = Button(ax_button, 'Pause')

    def toggle_pause(event):
        ui_state['paused'] = not ui_state['paused']
        if ui_state['paused']:
            button.label.set_text("Play")
            if ui_state['anim']:
                ui_state['anim'].event_source.stop()
        else:
            button.label.set_text("Pause")
            if ui_state['anim']:
                ui_state['anim'].event_source.start()

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
    print("  Beta Slider:       Adjust body shape (-10 to +10, uniform 'fatness')")
    print("  Override Checkbox: Enable to use slider betas, disable for original betas")

    anim = FuncAnimation(
        fig, on_frame,
        frames=range(total_frames),
        interval=interval,
        blit=False,
        repeat=True
    )
    ui_state['anim'] = anim

    # Cleanup on close
    def on_close(event):
        print("\nCleaning up renderer...")
        renderer.delete()

    fig.canvas.mpl_connect('close_event', on_close)

    plt.show()


if __name__ == '__main__':
    args = parse_arguments()
    setup_rendering_backend(args.gpu_id)
    main(args)
