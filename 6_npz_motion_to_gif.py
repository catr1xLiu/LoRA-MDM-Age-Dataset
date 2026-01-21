#!/usr/bin/env python3
"""
HumanML3D NPZ Motion Visualization Script

Loads HumanML3D-style NPZ files (from 3_export_humanml3d.py or data pipeline)
and visualizes the 3D motion with interactive controls.

Supports two visualization modes:
1. Root-centered: Shows motion relative to pelvis center (moonwalk effect)
2. Full trajectory: Reconstructs global motion using pelvis trajectory

Usage:
    # View root-centered motion (what the model learns)
    python3 6_npz_motion_to_gif.py --input data/humanml3d_joints_4/SUBJ01/SUBJ1_0_humanml3d_22joints.npz

    # View with full trajectory reconstruction
    python3 6_npz_motion_to_gif.py --input data/humanml3d_joints_4/SUBJ01/SUBJ1_0_humanml3d_22joints.npz --use-trajectory

    # Downsample for faster playback
    python3 6_npz_motion_to_gif.py --input path/to/motion.npz --use-trajectory --downsample 2

Controls:
    Frame Slider:   Scrub through animation frames
    Play/Pause:     Toggle animation playback
    Scroll Up/Down: Zoom in/out
    Spacebar:       Play/Pause toggle (future enhancement)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import argparse
import sys
from pathlib import Path


# HumanML3D 22-joint kinematic chain (skeleton structure)
# Joint order: 0:pelvis, 1:L_hip, 2:R_hip, 3:spine1, 4:L_knee, 5:R_knee, 6:spine2,
#              7:L_ankle, 8:R_ankle, 9:spine3, 10:L_foot, 11:R_foot, 12:neck, 13:L_collar,
#              14:R_collar, 15:head, 16:L_shoulder, 17:R_shoulder, 18:L_elbow, 19:R_elbow,
#              20:L_wrist, 21:R_wrist
KINEMATIC_CHAIN = [
    (0, 1), (1, 4), (4, 7), (7, 10),        # Left Leg
    (0, 2), (2, 5), (5, 8), (8, 11),        # Right Leg
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),  # Spine
    (9, 13), (13, 16), (16, 18), (18, 20),  # Left Arm
    (9, 14), (14, 17), (17, 19), (19, 21)   # Right Arm
]

# Joint names for labels
JOINT_NAMES = [
    'pelvis', 'L_hip', 'R_hip', 'spine1', 'L_knee', 'R_knee', 'spine2',
    'L_ankle', 'R_ankle', 'spine3', 'L_foot', 'R_foot', 'neck', 'L_collar',
    'R_collar', 'head', 'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow',
    'L_wrist', 'R_wrist'
]


def load_data(npz_path, use_trajectory):
    """
    Load and optionally reconstruct motion data from NPZ file.

    Args:
        npz_path: Path to the NPZ file
        use_trajectory: If True, reconstruct global motion using pelvis trajectory

    Returns:
        np.ndarray: Motion data (frames, joints, 3D coordinates)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required data arrays are missing
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"File not found: {npz_path}")

    try:
        data = np.load(npz_path)
    except Exception as e:
        raise ValueError(f"Failed to load NPZ file: {e}")

    if 'joints' not in data:
        raise ValueError(f"'joints' array not found in {npz_path}")

    root_centered_pose = data['joints']

    if use_trajectory:
        if 'pelvis_traj' not in data:
            raise ValueError("--use-trajectory specified but 'pelvis_traj' not found in file")

        pelvis_trajectory = data['pelvis_traj']

        # Ensure lengths match
        if root_centered_pose.shape[0] != pelvis_trajectory.shape[0]:
            print("Warning: Frame count mismatch between joints and trajectory. Trimming to shortest.")
            min_len = min(root_centered_pose.shape[0], pelvis_trajectory.shape[0])
            root_centered_pose = root_centered_pose[:min_len]
            pelvis_trajectory = pelvis_trajectory[:min_len]

        # Reconstruct motion: (T, 22, 3) + (T, 1, 3) -> (T, 22, 3)
        motion_data = root_centered_pose + pelvis_trajectory[:, None, :]
        print("Reconstructed motion with pelvis trajectory")
    else:
        motion_data = root_centered_pose
        print("Using root-centered 'joints' data (relative to pelvis)")

    return motion_data


def calculate_bounding_box_size(motion_data):
    """
    Calculate adaptive bounding box size based on motion extent.

    The box size is computed to fit the skeleton across all frames,
    with padding for comfortable viewing.

    Args:
        motion_data: Motion data array (frames, joints, 3)

    Returns:
        float: Bounding box size
    """
    max_extent = 0

    for frame_idx in range(motion_data.shape[0]):
        frame_points = motion_data[frame_idx]  # (joints, 3)

        # Filter NaN values if present
        valid_mask = ~np.isnan(frame_points).any(axis=1)
        if not np.any(valid_mask):
            continue

        valid_points = frame_points[valid_mask]
        center = np.mean(valid_points, axis=0)

        # Calculate max distance from center
        dists = np.abs(valid_points - center)
        max_dist = np.max(dists)
        if max_dist > max_extent:
            max_extent = max_dist

    # Add padding for comfortable viewing
    box_size = max_extent * 1.2
    return box_size


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize HumanML3D motion data from NPZ files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View root-centered motion
  python3 6_npz_motion_to_gif.py --input data/humanml3d_joints_4/SUBJ01/SUBJ1_0_humanml3d_22joints.npz

  # View with global trajectory
  python3 6_npz_motion_to_gif.py --input data/humanml3d_joints_4/SUBJ01/SUBJ1_0_humanml3d_22joints.npz --use-trajectory

  # Downsample frames for faster playback
  python3 6_npz_motion_to_gif.py --input motion.npz --downsample 2
        """
    )

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to input NPZ file'
    )

    parser.add_argument(
        '-t', '--use-trajectory',
        action='store_true',
        help='Reconstruct motion with pelvis trajectory (if available)'
    )

    parser.add_argument(
        '-d', '--downsample',
        type=int,
        default=1,
        help='Skip every Nth frame for faster playback (default: 1)'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        help='Playback frame rate in FPS (default: 20)'
    )

    parser.add_argument(
        '--show-labels',
        action='store_true',
        help='Display joint name labels (default: off for performance)'
    )

    return parser.parse_args()


def main():
    """Main interactive visualization loop."""
    args = parse_arguments()

    # Load data
    print(f"Loading NPZ file: {args.input}")
    try:
        motion_data = load_data(args.input, args.use_trajectory)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Downsample if requested
    motion_data = motion_data[::args.downsample]
    total_frames = motion_data.shape[0]

    print(f"Loaded {total_frames} frames")
    print("Controls: Frame slider to scrub, Play/Pause button, scroll to zoom")

    # Setup figure and axis
    fig = plt.figure(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)  # Space for widgets

    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15., azim=-90)
    ax.set_xlabel("X")
    ax.set_ylabel("Z (Forward)")
    ax.set_zlabel("Y (Up)")

    # Calculate bounding box
    box_size = calculate_bounding_box_size(motion_data)
    print(f"Bounding box size: {box_size:.4f}")

    # Initialize visual elements
    lines = [ax.plot([], [], [], 'r-', linewidth=1.5)[0] for _ in KINEMATIC_CHAIN]
    scatter = ax.scatter([], [], [], color='b', marker='.', s=20)

    # Initialize joint labels (if enabled)
    joint_labels = []
    if args.show_labels:
        joint_labels = [
            ax.text(0, 0, 0, name, fontsize=7, color='darkgreen', alpha=0.8,
                   horizontalalignment='left', verticalalignment='bottom')
            for name in JOINT_NAMES
        ]

    # UI state
    ui_state = {
        'frame': 0,
        'paused': False,
        'anim': None,
        'box_size': box_size,
    }

    # Scroll wheel zoom control
    def on_scroll(event):
        """Handle mouse scroll for zoom."""
        zoom_factor = 0.1
        if event.button == 'up':      # Scroll up -> zoom in
            ui_state['box_size'] *= (1 - zoom_factor)
        elif event.button == 'down':  # Scroll down -> zoom out
            ui_state['box_size'] *= (1 + zoom_factor)

        # Clamp to reasonable bounds
        ui_state['box_size'] = max(0.1, min(10.0, ui_state['box_size']))
        update_visuals(ui_state['frame'])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # Update visuals for a given frame
    def update_visuals(frame_index):
        """Render visualization for a specific frame."""
        frame_pose = motion_data[frame_index]  # (joints, 3)

        # Swap Y and Z for matplotlib (Y is up in data, Z is up in mplot3d)
        points = frame_pose[:, [0, 2, 1]]

        # Update joint positions
        scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

        # Update skeleton bones
        for line, (parent, child) in zip(lines, KINEMATIC_CHAIN):
            p1 = points[parent]
            p2 = points[child]
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])

        # Update joint labels
        for i, label in enumerate(joint_labels):
            label.set_position((points[i, 0], points[i, 1]))
            label.set_3d_properties(points[i, 2], zdir='z')

        # Update view bounds
        current_box_size = ui_state['box_size']
        center = np.mean(points, axis=0)
        ax.set_xlim(center[0] - current_box_size, center[0] + current_box_size)
        ax.set_ylim(center[1] - current_box_size, center[1] + current_box_size)
        ax.set_zlim(center[2] - current_box_size, center[2] + current_box_size)

        ax.set_title(f'Frame {frame_index * args.downsample} / {total_frames * args.downsample - 1}')

    # Animation frame callback
    def on_frame(frame):
        """Called each animation frame."""
        if ui_state['paused']:
            return lines + [scatter]

        ui_state['frame'] = frame
        frame_slider.eventson = False
        frame_slider.set_val(frame)
        frame_slider.eventson = True

        update_visuals(frame)
        return lines + [scatter]

    # Frame slider widget
    ax_slider = plt.axes([0.2, 0.10, 0.55, 0.03])
    frame_slider = Slider(ax_slider, 'Frame', 0, total_frames - 1, valinit=0, valfmt='%d')

    def update_slider(val):
        """Handle frame slider interaction."""
        frame = int(frame_slider.val)
        ui_state['frame'] = frame
        ui_state['paused'] = True
        button.label.set_text("Play")

        if ui_state['anim']:
            ui_state['anim'].event_source.stop()

        update_visuals(frame)
        fig.canvas.draw_idle()

    frame_slider.on_changed(update_slider)

    # Play/Pause button
    ax_button = plt.axes([0.8, 0.10, 0.1, 0.04])
    button = Button(ax_button, 'Pause')

    def toggle_pause(event):
        """Toggle animation playback."""
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
    print(f"Starting animation at {1000 // interval:.0f} FPS")

    anim = FuncAnimation(
        fig, on_frame,
        frames=range(total_frames),
        interval=interval,
        blit=False,
        repeat=True
    )
    ui_state['anim'] = anim

    plt.show()


if __name__ == "__main__":
    main()
