#!/usr/bin/env python3
"""
C3D Motion Capture Visualization Script

Visualizes motion capture data from C3D files with skeleton structure and joint labels.
The script reads marker positions and animates them in 3D space with interactive controls.

Usage:
    # Basic usage with default skeleton
    python3 inspect_file.py --file path/to/motion.c3d

    # Specify custom skeleton configuration
    python3 inspect_file.py --file path/to/motion.c3d --skeleton van_criekinge

    # Control animation playback
    python3 inspect_file.py --file path/to/motion.c3d --fps 30 --no-labels
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import numpy as np
from ezc3d import c3d


# Predefined skeleton structures for different datasets
SKELETON_DEFINITIONS = {
    'van_criekinge': [
        ("CentreOfMass", "TRXO"),
        ("TRXO", "HEDO"),  # Spine
        ("TRXO", "LCLO"),
        ("TRXO", "RCLO"),  # Shoulders
        ("LCLO", "LHUO"),
        ("LHUO", "LRAO"),
        ("LRAO", "LHNO"),  # Left Arm
        ("RCLO", "RHUO"),
        ("RHUO", "RRAO"),
        ("RRAO", "RHNO"),  # Right Arm
        ("CentreOfMass", "LFEP"),
        ("LFEP", "LFEO"),
        ("LFEO", "LTIO"),
        ("LTIO", "LFOO"),  # Left Leg
        ("CentreOfMass", "RFEP"),
        ("RFEP", "RFEO"),
        ("RFEO", "RTIO"),
        ("RTIO", "RFOO"),  # Right Leg
    ],
}


def load_c3d_file(file_path):
    """
    Load motion capture data from C3D file.

    Args:
        file_path: Path to the C3D file

    Returns:
        dict: C3D data structure with points and labels

    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file cannot be parsed as C3D
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"C3D file not found: {file_path}")

    try:
        c3d_data = c3d(file_path)
        return c3d_data
    except Exception as e:
        raise ValueError(f"Failed to parse C3D file: {e}")


def get_point_data(all_points, labels, label_name):
    """
    Extract 3D trajectory data for a specific marker label.

    Args:
        all_points: Motion data array (shape: [3, num_markers, num_frames])
        labels: List of marker label names
        label_name: Name of the marker to retrieve

    Returns:
        np.ndarray: Trajectory data for the marker or None if not found
    """
    if label_name in labels:
        idx = labels.index(label_name)
        return all_points[:, idx, :]
    return None


def validate_skeleton(c3d_data, skeleton):
    """
    Check if all markers in skeleton are present in the C3D data.

    Args:
        c3d_data: Loaded C3D data
        skeleton: List of (marker1, marker2) tuples

    Returns:
        list: Validated skeleton with only available connections
    """
    all_labels = c3d_data['parameters']['POINT']["LABELS"]["value"]
    validated_skeleton = []
    missing_markers = set()

    for p1_name, p2_name in skeleton:
        p1_exists = p1_name in all_labels
        p2_exists = p2_name in all_labels

        if p1_exists and p2_exists:
            validated_skeleton.append((p1_name, p2_name))
        else:
            if not p1_exists:
                missing_markers.add(p1_name)
            if not p2_exists:
                missing_markers.add(p2_name)

    if missing_markers:
        print(f"Warning: Some markers not found in C3D file: {missing_markers}")
        print(f"Using {len(validated_skeleton)} valid skeleton connections")

    return validated_skeleton


def create_visualization(c3d_data, skeleton, show_labels=True, fps=30):
    """
    Create interactive 3D visualization of motion capture data.

    Args:
        c3d_data: Loaded C3D data
        skeleton: List of (marker1, marker2) tuples defining skeleton structure
        show_labels: Whether to display marker labels
        fps: Frames per second for animation playback

    Returns:
        tuple: (figure, animation) for display
    """
    all_points = c3d_data['data']['points']
    all_labels = c3d_data['parameters']['POINT']["LABELS"]["value"]
    num_frames = all_points.shape[2]

    # Create figure and 3D axis with space for sliders
    fig = plt.figure(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.15)  # Space for sliders
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Motion Capture Data Visualization')

    # Extract line data for valid skeleton connections
    lines_data = []
    for p1_name, p2_name in skeleton:
        p1_data = get_point_data(all_points, all_labels, p1_name)
        p2_data = get_point_data(all_points, all_labels, p2_name)
        if p1_data is not None and p2_data is not None:
            lines_data.append((p1_data, p2_data))

    # Create line objects for skeleton
    lines = [ax.plot([], [], [], marker='o', color='blue', linewidth=2)[0] for _ in lines_data]

    # Create text labels for markers (if enabled)
    text_objects = {}
    if show_labels:
        unique_labels = set()
        for p1, p2 in skeleton:
            unique_labels.add(p1)
            unique_labels.add(p2)

        for label in unique_labels:
            label_data = get_point_data(all_points, all_labels, label)
            if label_data is not None:
                text_objects[label] = ax.text(0, 0, 0, label, color='black', fontsize=8)

    # Calculate view bounds from data
    max_extent = _calculate_max_extent(all_points)
    box_size = max_extent * 0.1

    # Setup interactive zoom
    zoom_factor = [1.0]  # Use list to allow modification in nested function

    def on_scroll(event):
        """Handle mouse scroll for zoom control."""
        if event.button == 'up':
            zoom_factor[0] /= 1.1
        elif event.button == 'down':
            zoom_factor[0] *= 1.1

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # UI state for playback control
    ui_state = {
        'frame': 0,
        'paused': False,
        'anim': None,
    }

    # Animation update function
    def update(frame):
        """Update visualization for current frame."""
        # If paused and frame hasn't changed, don't update
        if ui_state['paused'] and frame == ui_state['frame']:
            return lines + list(text_objects.values())

        current_points = []

        # Update skeleton lines
        for line, (p1_data, p2_data) in zip(lines, lines_data):
            xs = [p1_data[0, frame], p2_data[0, frame]]
            ys = [p1_data[1, frame], p2_data[1, frame]]
            zs = [p1_data[2, frame], p2_data[2, frame]]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)

            # Collect points for center calculation
            current_points.append([p1_data[0, frame], p1_data[1, frame], p1_data[2, frame]])
            current_points.append([p2_data[0, frame], p2_data[1, frame], p2_data[2, frame]])

        # Update marker labels
        for label, text_obj in text_objects.items():
            label_data = get_point_data(all_points, all_labels, label)
            if label_data is not None:
                x, y, z = label_data[0, frame], label_data[1, frame], label_data[2, frame]
                if not np.isnan(x):
                    text_obj.set_position((x, y))
                    text_obj.set_3d_properties(z)
                    text_obj.set_visible(True)
                else:
                    text_obj.set_visible(False)

        # Update camera view to follow motion
        if current_points:
            current_points_array = np.array(current_points)
            valid_mask = ~np.isnan(current_points_array[:, 0])
            if np.any(valid_mask):
                valid_points = current_points_array[valid_mask]
                center = np.mean(valid_points, axis=0)

                current_box_size = box_size * zoom_factor[0]
                ax.set_xlim(center[0] - current_box_size, center[0] + current_box_size)
                ax.set_ylim(center[1] - current_box_size, center[1] + current_box_size)
                ax.set_zlim(center[2] - current_box_size, center[2] + current_box_size)

        # Update title with current frame info
        ax.set_title(f'Frame {frame} / {num_frames - 1}')

        return lines + list(text_objects.values())

    # Animation frame callback
    def on_frame(frame):
        """Callback for animation frame updates."""
        if ui_state['paused']:
            return lines + list(text_objects.values())

        ui_state['frame'] = frame
        frame_slider.eventson = False
        frame_slider.set_val(frame)
        frame_slider.eventson = True

        return update(frame)

    # Frame slider widget
    ax_frame_slider = plt.axes([0.2, 0.08, 0.55, 0.03])
    frame_slider = Slider(ax_frame_slider, 'Frame', 0, num_frames - 1, valinit=0, valfmt='%d')

    def update_frame_slider(val):
        """Handle frame slider changes."""
        frame = int(frame_slider.val)
        ui_state['frame'] = frame
        ui_state['paused'] = True
        button.label.set_text("Play")
        if ui_state['anim']:
            ui_state['anim'].event_source.stop()

        # Update all visual elements directly
        current_points = []

        # Update skeleton lines
        for line, (p1_data, p2_data) in zip(lines, lines_data):
            xs = [p1_data[0, frame], p2_data[0, frame]]
            ys = [p1_data[1, frame], p2_data[1, frame]]
            zs = [p1_data[2, frame], p2_data[2, frame]]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)

            # Collect points for center calculation
            current_points.append([p1_data[0, frame], p1_data[1, frame], p1_data[2, frame]])
            current_points.append([p2_data[0, frame], p2_data[1, frame], p2_data[2, frame]])

        # Update marker labels
        for label, text_obj in text_objects.items():
            label_data = get_point_data(all_points, all_labels, label)
            if label_data is not None:
                x, y, z = label_data[0, frame], label_data[1, frame], label_data[2, frame]
                if not np.isnan(x):
                    text_obj.set_position((x, y))
                    text_obj.set_3d_properties(z)
                    text_obj.set_visible(True)
                else:
                    text_obj.set_visible(False)

        # Update camera view to follow motion
        if current_points:
            current_points_array = np.array(current_points)
            valid_mask = ~np.isnan(current_points_array[:, 0])
            if np.any(valid_mask):
                valid_points = current_points_array[valid_mask]
                center = np.mean(valid_points, axis=0)

                current_box_size = box_size * zoom_factor[0]
                ax.set_xlim(center[0] - current_box_size, center[0] + current_box_size)
                ax.set_ylim(center[1] - current_box_size, center[1] + current_box_size)
                ax.set_zlim(center[2] - current_box_size, center[2] + current_box_size)

        # Update title with current frame info
        ax.set_title(f'Frame {frame} / {num_frames - 1}')

        fig.canvas.draw_idle()

    frame_slider.on_changed(update_frame_slider)

    # Play/Pause button
    ax_button = plt.axes([0.8, 0.08, 0.1, 0.04])
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
    interval_ms = int(1000 / fps)  # Convert FPS to milliseconds
    ani = animation.FuncAnimation(
        fig, on_frame, frames=num_frames, interval=interval_ms, blit=False
    )
    ui_state['anim'] = ani

    return fig, ani


def _calculate_max_extent(all_points):
    """
    Calculate the maximum extent of motion in the dataset.

    Args:
        all_points: Motion data array

    Returns:
        float: Maximum distance from center across all frames
    """
    max_extent = 0
    for frame in range(all_points.shape[2]):
        frame_points = all_points[:, :, frame]

        # Filter out NaNs
        valid_mask = ~np.isnan(frame_points[0])
        if not np.any(valid_mask):
            continue

        valid_points = frame_points[:, valid_mask]
        center = np.mean(valid_points, axis=1)

        # Calculate max distance from center
        dists = np.abs(valid_points - center[:, np.newaxis])
        max_dist = np.max(dists)
        if max_dist > max_extent:
            max_extent = max_dist

    return max_extent


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize motion capture data from C3D files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a C3D file with default skeleton
  python3 inspect_file.py --file motion.c3d

  # Use Van Criekinge skeleton
  python3 inspect_file.py --file motion.c3d --skeleton van_criekinge

  # Disable marker labels and set playback speed
  python3 inspect_file.py --file motion.c3d --no-labels --fps 60
        """,
    )

    parser.add_argument(
        '-f', '--file',
        type=str,
        required=True,
        help='Path to the C3D motion capture file'
    )

    parser.add_argument(
        '-s', '--skeleton',
        type=str,
        default='van_criekinge',
        choices=list(SKELETON_DEFINITIONS.keys()),
        help='Skeleton structure to use (default: van_criekinge)'
    )

    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Do not display marker labels'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Animation playback speed in frames per second (default: 30)'
    )

    parser.add_argument(
        '--list-markers',
        action='store_true',
        help='Print all available markers in the file and exit'
    )

    return parser.parse_args()


def main():
    """Main visualization pipeline."""
    args = parse_arguments()

    # Load C3D file
    try:
        print(f"Loading C3D file: {args.file}")
        c3d_data = load_c3d_file(args.file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Get available markers
    all_labels = c3d_data['parameters']['POINT']["LABELS"]["value"]
    all_points = c3d_data['data']['points']

    print(f"  Found {len(all_labels)} markers")
    print(f"  Total frames: {all_points.shape[2]}")

    # Handle --list-markers option
    if args.list_markers:
        print("\nAvailable markers:")
        for i, label in enumerate(all_labels):
            print(f"  [{i:2d}] {label}")
        return

    # Get skeleton definition
    if args.skeleton not in SKELETON_DEFINITIONS:
        print(f"Error: Unknown skeleton '{args.skeleton}'")
        print(f"Available skeletons: {', '.join(SKELETON_DEFINITIONS.keys())}")
        sys.exit(1)

    skeleton = SKELETON_DEFINITIONS[args.skeleton]

    # Validate skeleton
    skeleton = validate_skeleton(c3d_data, skeleton)
    if not skeleton:
        print("Error: No valid skeleton connections found")
        sys.exit(1)

    # Create and display visualization
    print("\nCreating visualization...")
    print("  Controls: Scroll wheel to zoom, close window to exit")

    fig, ani = create_visualization(
        c3d_data,
        skeleton,
        show_labels=not args.no_labels,
        fps=args.fps
    )

    plt.show()


if __name__ == '__main__':
    main()
