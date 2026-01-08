#!/usr/bin/env python3
"""View SMPL parameters from npz file in an interactive 3D viewer."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import argparse
import torch
import smplx

# SMPL joint configuration
SMPL_JOINTS = [
    "pelvis", "L_hip", "R_hip", "spine1", "L_knee", "R_knee", "spine2",
    "L_ankle", "R_ankle", "spine3", "L_foot", "R_foot", "neck",
    "L_collar", "R_collar", "head", "L_shoulder", "R_shoulder",
    "L_elbow", "R_elbow", "L_wrist", "R_wrist", "L_hand", "R_hand"
]

# Kinematic chain (bones to draw)
KINEMATIC_CHAIN = [
    (0, 1), (1, 4), (4, 7), (7, 10),  # Left Leg
    (0, 2), (2, 5), (5, 8), (8, 11),  # Right Leg
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), # Spine
    (9, 13), (13, 16), (16, 18), (18, 20), (18, 22), # Left Arm
    (9, 14), (14, 17), (17, 19), (19, 21), (19, 23)  # Right Arm
]

def load_smpl_params(npz_path):
    """Load SMPL parameters from npz file."""
    data = np.load(npz_path, allow_pickle=True)
    
    print("Keys in npz file:", list(data.keys()))
    
    poses = data['poses']  # (T, 72) - global_orient (3) + body_pose (69)
    trans = data['trans']  # (T, 3)
    betas = data['betas']  # (10,)
    gender = str(data['gender']) if 'gender' in data else 'neutral'
    
    print(f"Loaded: {poses.shape[0]} frames")
    print(f"Gender: {gender}")
    print(f"Betas shape: {betas.shape}")
    
    return poses, trans, betas, gender

def generate_joints_from_smpl(smpl, poses, trans, betas):
    """Generate joint positions from SMPL parameters."""
    T = poses.shape[0]
    
    # Split poses into global_orient and body_pose
    global_orient = poses[:, :3]  # (T, 3)
    body_pose = poses[:, 3:]      # (T, 69)
    
    # Convert to torch tensors
    global_orient_t = torch.from_numpy(global_orient).float()
    body_pose_t = torch.from_numpy(body_pose).float()
    trans_t = torch.from_numpy(trans).float()
    betas_t = torch.from_numpy(betas).float().unsqueeze(0).expand(T, -1)
    
    # Forward pass
    with torch.no_grad():
        output = smpl(
            global_orient=global_orient_t,
            body_pose=body_pose_t,
            betas=betas_t,
            transl=trans_t,
            pose2rot=True
        )
        joints = output.joints[:, :24, :].cpu().numpy()  # (T, 24, 3)
    
    return joints

def calculate_bounding_box_size(joints):
    """Calculate a fixed bounding box size based on max extent."""
    max_extent = 0
    for frame_idx in range(joints.shape[0]):
        frame_points = joints[frame_idx]
        valid_mask = ~np.isnan(frame_points).any(axis=1)
        if not np.any(valid_mask):
            continue
        valid_points = frame_points[valid_mask]
        center = np.mean(valid_points, axis=0)
        dists = np.abs(valid_points - center)
        max_dist = np.max(dists)
        if max_dist > max_extent:
            max_extent = max_dist
    
    return max_extent * 1.2

def main():
    parser = argparse.ArgumentParser(description="Visualize SMPL parameters.")
    parser.add_argument("--input", type=str, required=True, help="Path to SMPL params npz file")
    parser.add_argument("--model_dir", type=str, default=".", help="Path to SMPL models directory")
    parser.add_argument("--downsample", type=int, default=1, help="Plot every Nth frame")
    args = parser.parse_args()
    
    # Load SMPL parameters
    print(f"Loading SMPL parameters from {args.input}...")
    poses, trans, betas, gender = load_smpl_params(args.input)
    
    # Create SMPL model
    print(f"Creating SMPL model (gender={gender})...")
    smpl = smplx.create(
        model_path=args.model_dir,
        model_type="smpl",
        gender=gender,
        use_pca=False,
        num_betas=10
    )
    smpl.eval()
    
    # Generate joint positions
    print("Generating joint positions...")
    joints = generate_joints_from_smpl(smpl, poses, trans, betas)
    
    # Downsample
    joints = joints[::args.downsample]
    total_frames = joints.shape[0]
    print(f"Total frames (after downsample): {total_frames}")
    
    # Setup plot
    fig = plt.figure(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=15., azim=-90)
    
    # Calculate bounding box
    box_size = calculate_bounding_box_size(joints)
    print(f"Bounding box size: {box_size:.4f}")
    
    # Initialize lines and scatter
    lines = [ax.plot([], [], [], 'r-', linewidth=1.5)[0] for _ in KINEMATIC_CHAIN]
    scatter = ax.scatter([], [], [], color='b', marker='o', s=30)
    
    # Joint labels
    joint_labels = [ax.text(0, 0, 0, name, fontsize=7, color='darkgreen', alpha=0.8,
                            horizontalalignment='left', verticalalignment='bottom')
                    for name in SMPL_JOINTS]
    
    ax.set_xlabel("X")
    ax.set_ylabel("Z (Forward)")
    ax.set_zlabel("Y (Up)")
    
    # UI state
    ui_state = {
        'frame': 0,
        'paused': False,
        'anim': None,
        'box_size': box_size
    }
    
    # Scroll wheel zoom
    def on_scroll(event):
        zoom_factor = 0.1
        if event.button == 'up':
            ui_state['box_size'] *= (1 - zoom_factor)
        elif event.button == 'down':
            ui_state['box_size'] *= (1 + zoom_factor)
        ui_state['box_size'] = max(0.1, min(10.0, ui_state['box_size']))
        update_visuals(ui_state['frame'])
        fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    # Update visuals
    def update_visuals(frame_index):
        frame_pose = joints[frame_index]  # (24, 3)
        
        # Swap Y and Z for plotting (Y is UP in SMPL, Z is UP in matplotlib 3D)
        points = frame_pose[:, [0, 2, 1]]
        
        # Update joints
        scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])
        
        # Update bones
        for line, (parent, child) in zip(lines, KINEMATIC_CHAIN):
            p1 = points[parent]
            p2 = points[child]
            line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            line.set_3d_properties([p1[2], p2[2]])
        
        # Update labels
        for i, label in enumerate(joint_labels):
            label.set_position((points[i, 0], points[i, 1]))
            label.set_3d_properties(points[i, 2], zdir='z')
        
        # Update camera
        current_box_size = ui_state['box_size']
        center = np.mean(points, axis=0)
        ax.set_xlim(center[0] - current_box_size, center[0] + current_box_size)
        ax.set_ylim(center[1] - current_box_size, center[1] + current_box_size)
        ax.set_zlim(center[2] - current_box_size, center[2] + current_box_size)
        
        ax.set_title(f'Frame {frame_index * args.downsample} (Scroll to zoom)')
    
    # Animation frame update
    def on_frame(frame):
        if ui_state['paused']:
            return lines + [scatter]
        
        ui_state['frame'] = frame
        slider.eventson = False
        slider.set_val(frame)
        slider.eventson = True
        
        update_visuals(frame)
        return lines + [scatter]
    
    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, total_frames - 1, valinit=0, valfmt='%d')
    
    def update_slider(val):
        frame = int(slider.val)
        ui_state['frame'] = frame
        ui_state['paused'] = True
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
    
    # Create animation
    print(f"Starting interactive animation...")
    interval = 50 * args.downsample
    anim = FuncAnimation(fig, on_frame, frames=range(total_frames), interval=interval, blit=False, repeat=True)
    ui_state['anim'] = anim
    
    plt.show()

if __name__ == "__main__":
    main()
