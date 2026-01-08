#!/usr/bin/env python3
"""
Loads a HumanML3D-style NPZ file (from 3_export_humanml3d.py) 
and reconstructs the 3D motion as a GIF.

Use this to visualize the difference between the root-centered data
and the fully reconstructed data.

Example:

# 1. To see the "moonwalk" (what the model was learning from):
python reconstruct_gif.py --input data/humanml3d/SUBJ01/SUBJ1_0_humanml3d_22joints.npz --output moonwalk.gif

# 2. To see the *correct* motion with the trajectory:
python reconstruct_gif.py --input data/humanml3d/SUBJ01/SUBJ1_0_humanml3d_22joints.npz --output correct_walk.gif --use_trajectory
"""

# python 6_npz_motion_to_gif.py --input Comp_v6_KLD01/SUBJ01/SUBJ1_1_humanml3d_22joints.npz --use_trajectory

# python 6_npz_motion_to_gif.py --input Comp_v6_KLD01/SUBJ01/SUBJ1_1_humanml3d_22joints.npz --use_trajectory

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import sys
from tqdm import tqdm

# Define the kinematic chain (skeleton) for the 22-joint HumanML3D skeleton
# This is hardcoded to avoid external dependencies.
# Joint names: 0:pelvis, 1:L_hip, 2:R_hip, 3:spine1, 4:L_knee, 5:R_knee, 6:spine2, 
# 7:L_ankle, 8:R_ankle, 9:spine3, 10:L_foot, 11:R_foot, 12:neck, 13:L_collar, 
# 14:R_collar, 15:head, 16:L_shoulder, 17:R_shoulder, 18:L_elbow, 19:R_elbow, 
# 20:L_wrist, 21:R_wrist
KINEMATIC_CHAIN = [
    (0, 1), (1, 4), (4, 7), (7, 10),  # Left Leg
    (0, 2), (2, 5), (5, 8), (8, 11),  # Right Leg
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), # Spine
    (9, 13), (13, 16), (16, 18), (18, 20), # Left Arm
    (9, 14), (14, 17), (17, 19), (19, 21)  # Right Arm
]

# Joint names for the 22-joint HumanML3D skeleton
JOINT_NAMES = [
    'pelvis', 'L_hip', 'R_hip', 'spine1', 'L_knee', 'R_knee', 'spine2',
    'L_ankle', 'R_ankle', 'spine3', 'L_foot', 'R_foot', 'neck', 'L_collar',
    'R_collar', 'head', 'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow',
    'L_wrist', 'R_wrist'
]

def load_data(npz_path, use_trajectory):
    """Loads motion data and optionally reconstructs it with the pelvis trajectory."""
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"Error: Could not load file {npz_path}")
        print(e)
        sys.exit(1)
        
    if 'joints' not in data:
        print(f"Error: 'joints' array not found in {npz_path}")
        sys.exit(1)
        
    root_centered_pose = data['joints']
    
    if use_trajectory:
        if 'pelvis_traj' not in data:
            print(f"Error: --use_trajectory was specified, but 'pelvis_traj' array not found.")
            sys.exit(1)
        
        pelvis_trajectory = data['pelvis_traj']
        
        # Ensure lengths match
        if root_centered_pose.shape[0] != pelvis_trajectory.shape[0]:
            print(f"Warning: Mismatch in frame count. Trimming to shortest.")
            min_len = min(root_centered_pose.shape[0], pelvis_trajectory.shape[0])
            root_centered_pose = root_centered_pose[:min_len]
            pelvis_trajectory = pelvis_trajectory[:min_len]

        # Reconstruct by adding the trajectory back to the root-centered pose
        # (T, 22, 3) + (T, 1, 3) -> (T, 22, 3)
        motion_data = root_centered_pose + pelvis_trajectory[:, None, :]
        print("Reconstructed motion with pelvis trajectory.")
    else:
        motion_data = root_centered_pose
        print("Using root-centered 'joints' data (moonwalk).")
        
    return motion_data

def calculate_bounding_box_size(motion_data):
    """
    Calculates a fixed bounding box size based on the maximum extent of the character
    relative to its center across all frames.
    """
    # motion_data: (T, J, 3)
    # We want a box big enough to fit the skeleton at any frame, centered on the skeleton.
    
    max_extent = 0
    for frame_idx in range(motion_data.shape[0]):
        frame_points = motion_data[frame_idx] # (J, 3)
        
        # Filter out NaNs if any (though HumanML3D shouldn't have them usually)
        valid_mask = ~np.isnan(frame_points).any(axis=1)
        if not np.any(valid_mask):
            continue
            
        valid_points = frame_points[valid_mask]
        center = np.mean(valid_points, axis=0)
        
        # Calculate max distance from center in any dimension
        dists = np.abs(valid_points - center) # (J, 3)
        max_dist = np.max(dists)
        if max_dist > max_extent:
            max_extent = max_dist

    # Add a little padding
    box_size = max_extent * 1.2 
    return box_size

from matplotlib.widgets import Slider, Button

def main():
    parser = argparse.ArgumentParser(description="Visualize HumanML3D NPZ file interactively.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .npz file")
    parser.add_argument("--use_trajectory", action="store_true", help="Add the 'pelvis_traj' to the 'joints' data.")
    parser.add_argument("--downsample", type=int, default=1, help="Plot every Nth frame (default: 1)")
    args = parser.parse_args()

    # 1. Load and (optionally) reconstruct the data
    motion_data = load_data(args.input, args.use_trajectory)
    
    # Downsample frames
    motion_data = motion_data[::args.downsample]
    total_frames = motion_data.shape[0]

    # 2. Setup the plot
    fig = plt.figure(figsize=(10, 8))
    # Adjust layout to make room for widgets at the bottom
    plt.subplots_adjust(bottom=0.2)
    
    ax = fig.add_subplot(111, projection='3d')
    
    # Initial view
    ax.view_init(elev=15., azim=-90) 

    # Calculate fixed bounding box size
    box_size = calculate_bounding_box_size(motion_data)
    print(f"Calculated bounding box size: {box_size:.4f}")

    # Initialize lines and scatter
    lines = [ax.plot([], [], [], 'r-', linewidth=1.5)[0] for _ in KINEMATIC_CHAIN]
    scatter = ax.scatter([], [], [], color='b', marker='.')
    
    # Initialize joint name text labels
    joint_labels = [ax.text(0, 0, 0, name, fontsize=7, color='darkgreen', alpha=0.8,
                            horizontalalignment='left', verticalalignment='bottom') 
                    for name in JOINT_NAMES]

    ax.set_xlabel("X")
    ax.set_ylabel("Z (Forward)")
    ax.set_zlabel("Y (Up)")

    # --- UI State ---
    ui_state = {
        'frame': 0,
        'paused': False,
        'anim': None,  # Will hold the animation object
        'box_size': box_size  # For zoom functionality
    }

    # Scroll wheel zoom handler
    def on_scroll(event):
        zoom_factor = 0.1
        if event.button == 'up':  # Scroll up -> zoom in
            ui_state['box_size'] *= (1 - zoom_factor)
        elif event.button == 'down':  # Scroll down -> zoom out
            ui_state['box_size'] *= (1 + zoom_factor)
        # Clamp the box size to reasonable bounds
        ui_state['box_size'] = max(0.1, min(10.0, ui_state['box_size']))
        # Update the view immediately
        update_visuals(ui_state['frame'])
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # 3. Define the update function (visuals only)
    def update_visuals(frame_index):
        # Get the pose for this frame
        frame_pose = motion_data[frame_index] # (J, 3) -> (X, Y, Z)
        
        # Swap Y and Z for plotting (Y is UP in data, Z is UP in mplot3d)
        points = frame_pose[:, [0, 2, 1]] 

        # Update joints
        scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

        # Update bones
        for line, (parent, child) in zip(lines, KINEMATIC_CHAIN):
            p1 = points[parent]
            p2 = points[child]
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])
        
        # Update joint labels
        for i, label in enumerate(joint_labels):
            label.set_position((points[i, 0], points[i, 1]))
            label.set_3d_properties(points[i, 2], zdir='z')
            
        # Update camera center using current box_size
        current_box_size = ui_state['box_size']
        center = np.mean(points, axis=0)
        ax.set_xlim(center[0] - current_box_size, center[0] + current_box_size)
        ax.set_ylim(center[1] - current_box_size, center[1] + current_box_size)
        ax.set_zlim(center[2] - current_box_size, center[2] + current_box_size)
        
        ax.set_title(f'Frame {frame_index * args.downsample} (Scroll to zoom)')

    # 4. Animation Frame Update
    def on_frame(frame):
        if ui_state['paused']:
            # If paused, keep sending the current slider value
            # This prevents jumpiness if the background anim is still trying to run,
            # though usually event_source.stop() handles true pausing.
            return lines + [scatter]
            
        ui_state['frame'] = frame
        slider.eventson = False # Disable event calculation to avoid recursive loop
        slider.set_val(frame)
        slider.eventson = True
        
        update_visuals(frame)
        return lines + [scatter]

    # 5. UI Widgets
    
    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, total_frames - 1, valinit=0, valfmt='%d')

    def update_slider(val):
        # User interaction with slider
        frame = int(slider.val)
        ui_state['frame'] = frame
        ui_state['paused'] = True # Auto-pause when dragging
        button.label.set_text("Play")
        if ui_state['anim']:
            ui_state['anim'].event_source.stop()
        
        update_visuals(frame)
        fig.canvas.draw_idle()

    slider.on_changed(update_slider)

    # Button
    ax_button = plt.axes([0.85, 0.05, 0.1, 0.04])
    button = Button(ax_button, 'Pause')

    def toggle_pause(event):
        ui_state['paused'] = not ui_state['paused']
        if ui_state['paused']:
            button.label.set_text("Play")
            ui_state['anim'].event_source.stop()
        else:
            button.label.set_text("Pause")
            ui_state['anim'].event_source.start()

    button.on_clicked(toggle_pause)

    # 6. Create and show the animation
    print(f"Starting interactive animation with {total_frames} frames...")
    interval = 50 * args.downsample 
    
    # NOTE: frames=range(...) is important so it loops correctly with proper indices
    anim = FuncAnimation(fig, on_frame, frames=range(total_frames), interval=interval, blit=False, repeat=True)
    ui_state['anim'] = anim
    
    plt.show()

if __name__ == "__main__":
    main()