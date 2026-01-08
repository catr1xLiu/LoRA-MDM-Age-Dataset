#!/usr/bin/env python3
"""Debug script to check SMPL shoulder joint positions."""

import torch
import smplx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Model directory (update this path as needed)
MODEL_DIR = "."

# SMPL joint indices
SMPL_JOINTS = [
    "pelvis", "L_hip", "R_hip", "spine1", "L_knee", "R_knee", "spine2",
    "L_ankle", "R_ankle", "spine3", "L_foot", "R_foot", "neck",
    "L_collar", "R_collar", "head", "L_shoulder", "R_shoulder",
    "L_elbow", "R_elbow", "L_wrist", "R_wrist", "L_hand", "R_hand"
]
J = {n: i for i, n in enumerate(SMPL_JOINTS)}

# Create SMPL model
print("Loading SMPL model...")
smpl = smplx.create(
    model_path=MODEL_DIR,
    model_type="smpl",
    gender="neutral",
    use_pca=False,
    num_betas=10
)
smpl.eval()

# Generate T-pose (zero pose)
print("Generating T-pose...")
with torch.no_grad():
    out = smpl(
        global_orient=torch.zeros(1, 3),
        body_pose=torch.zeros(1, 69),
        betas=torch.zeros(1, 10),
        transl=torch.zeros(1, 3)
    )
    joints = out.joints[0].cpu().numpy()  # (24,3)

# Extract relevant joints
L_collar = joints[J["L_collar"]]
R_collar = joints[J["R_collar"]]
L_sh = joints[J["L_shoulder"]]
R_sh = joints[J["R_shoulder"]]

print("\n" + "="*60)
print("SMPL Joint Positions (T-pose)")
print("="*60)
print(f"L_collar:    {L_collar}")
print(f"R_collar:    {R_collar}")
print(f"L_shoulder:  {L_sh}")
print(f"R_shoulder:  {R_sh}")
print("="*60)

# Calculate differences
print("\nDifferences:")
print(f"L_collar - R_collar:    {L_collar - R_collar}")
print(f"L_shoulder - R_shoulder: {L_sh - R_sh}")

# Print Z-axis comparison
print("\n" + "="*60)
print("Z-AXIS COMPARISON (Forward)")
print("="*60)
print(f"L_shoulder[Z]: {L_sh[2]:.6f}")
print(f"R_shoulder[Z]: {R_sh[2]:.6f}")
print(f"Difference:    {abs(L_sh[2] - R_sh[2]):.6f}")

if abs(L_sh[2] - R_sh[2]) > 1e-5:
    print("⚠️  WARNING: L_shoulder and R_shoulder have different Z coordinates!")
    print("   The SMPL skeleton is NOT symmetric in the forward direction.")
else:
    print("✅ L_shoulder and R_shoulder have the same Z coordinate (symmetric)")
print("="*60)

# Coordinate system note
print("\nSMPL Coordinate System:")
print("  X: right/left")
print("  Y: up/down")
print("  Z: forward/backward")

# Visualize the skeleton
print("\nGenerating 3D visualization...")
fig = plt.figure(figsize=(12, 5))

# Define some bones to draw
BONES = [
    (J["pelvis"], J["spine1"]),
    (J["spine1"], J["spine2"]),
    (J["spine2"], J["spine3"]),
    (J["spine3"], J["neck"]),
    (J["neck"], J["head"]),
    (J["spine3"], J["L_collar"]),
    (J["spine3"], J["R_collar"]),
    (J["L_collar"], J["L_shoulder"]),
    (J["R_collar"], J["R_shoulder"]),
    (J["L_shoulder"], J["L_elbow"]),
    (J["R_shoulder"], J["R_elbow"]),
    (J["L_elbow"], J["L_wrist"]),
    (J["R_elbow"], J["R_wrist"]),
]

# Three views
views = [
    (0, 0, "Front View (Y-Z)"),
    (0, 90, "Side View (X-Y)"),
    (90, 0, "Top View (X-Z)")
]

for i, (elev, azim, title) in enumerate(views):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    
    # Draw bones
    for parent, child in BONES:
        p1 = joints[parent]
        p2 = joints[child]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'b-', linewidth=2)
    
    # Highlight shoulder and collar joints
    ax.scatter(*L_collar, color='red', s=100, marker='o', label='L_collar')
    ax.scatter(*R_collar, color='blue', s=100, marker='o', label='R_collar')
    ax.scatter(*L_sh, color='darkred', s=100, marker='s', label='L_shoulder')
    ax.scatter(*R_sh, color='darkblue', s=100, marker='s', label='R_shoulder')
    
    ax.set_xlabel('X (left/right)')
    ax.set_ylabel('Y (up/down)')
    ax.set_zlabel('Z (forward/back)')
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    
    # Equal aspect ratio
    max_range = 0.6
    mid_x = 0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(-0.2, max_range * 2 - 0.2)
    ax.set_zlim(mid_x - max_range, mid_x + max_range)
    
    if i == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('smpl_shoulder_debug.png', dpi=150)
print("Saved visualization to 'smpl_shoulder_debug.png'")
plt.show()
