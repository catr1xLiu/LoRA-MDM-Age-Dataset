"""
Simple SMPL model implementation that loads from .npz files.
Bypasses the need for chumpy by loading data directly.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


def batch_rodrigues(rot_vecs):
    """
    Convert axis-angle rotations to rotation matrices.

    Args:
        rot_vecs: Rotation vectors of shape (batch_size, 3)

    Returns:
        Rotation matrices of shape (batch_size, 3, 3)
    """
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.cos(angle)
    sin = torch.sin(angle)

    # Compute rotation matrix using Rodrigues' formula
    rx, ry, rz = rot_dir[:, 0], rot_dir[:, 1], rot_dir[:, 2]
    zeros = torch.zeros(batch_size, dtype=dtype, device=device)

    K = torch.stack([
        torch.stack([zeros, -rz, ry], dim=1),
        torch.stack([rz, zeros, -rx], dim=1),
        torch.stack([-ry, rx, zeros], dim=1)
    ], dim=1)

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    # Reshape sin and cos for proper broadcasting: (batch_size,) -> (batch_size, 1, 1)
    sin = sin.view(batch_size, 1, 1)
    cos = cos.view(batch_size, 1, 1)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    return rot_mat


class SimpleSMPL:
    """Simplified SMPL model that loads from .npz files."""

    def __init__(self, model_path: str, gender: str = 'neutral'):
        """
        Initialize SMPL model from .npz file.

        Args:
            model_path: Path to the SMPL model .npz file
            gender: Gender of the model ('neutral', 'male', or 'female')
        """
        # Load model data
        model_data = np.load(model_path)

        # Extract model parameters
        self.v_template = torch.from_numpy(model_data['v_template'].astype(np.float32))
        self.shapedirs = torch.from_numpy(model_data['shapedirs'].astype(np.float32))
        self.posedirs = torch.from_numpy(model_data['posedirs'].astype(np.float32))

        # Handle J_regressor - might be sparse or dense
        j_reg = model_data['J_regressor']
        if hasattr(j_reg, 'toarray'):
            j_reg = j_reg.toarray()
        self.J_regressor = torch.from_numpy(j_reg.astype(np.float32))

        self.weights = torch.from_numpy(model_data['weights'].astype(np.float32))
        self.faces = model_data['f'].astype(np.int32)
        self.kintree_table = model_data['kintree_table']

        # Model dimensions
        self.num_betas = self.shapedirs.shape[-1]
        self.num_joints = self.J_regressor.shape[0]

        # Parent indices for kinematic tree
        self.parents = torch.tensor([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
                                    dtype=torch.long)

    def forward(self,
                betas: Optional[torch.Tensor] = None,
                global_orient: Optional[torch.Tensor] = None,
                body_pose: Optional[torch.Tensor] = None,
                transl: Optional[torch.Tensor] = None):
        """
        Forward pass of SMPL model.

        Args:
            betas: Shape parameters (batch_size, num_betas)
            global_orient: Global rotation (batch_size, 3)
            body_pose: Body pose parameters (batch_size, 69)
            transl: Translation (batch_size, 3)

        Returns:
            Dictionary with 'vertices' key containing the mesh vertices
        """
        batch_size = 1 if betas is None else betas.shape[0]
        device = betas.device if betas is not None else torch.device('cpu')

        # Move tensors to the correct device
        v_template = self.v_template.to(device)
        shapedirs = self.shapedirs.to(device)
        J_regressor = self.J_regressor.to(device)
        weights = self.weights.to(device)
        parents = self.parents.to(device)

        # Initialize shape-dependent template
        if betas is None:
            betas = torch.zeros((batch_size, self.num_betas), device=device)

        # Trim betas to match shapedirs if necessary
        num_shape_params = shapedirs.shape[-1]
        if betas.shape[1] < num_shape_params:
            # Pad betas with zeros
            padding = torch.zeros((betas.shape[0], num_shape_params - betas.shape[1]), device=device)
            betas = torch.cat([betas, padding], dim=1)
        elif betas.shape[1] > num_shape_params:
            # Trim betas
            betas = betas[:, :num_shape_params]

        # Apply shape blend shapes
        # shapedirs is (num_verts, 3, num_betas)
        v_shaped = v_template + torch.einsum('bl,mkl->bmk', betas, shapedirs)

        # Initialize poses
        if global_orient is None:
            global_orient = torch.zeros((batch_size, 3), device=device)
        if body_pose is None:
            body_pose = torch.zeros((batch_size, 69), device=device)

        # Concatenate global orientation and body pose
        full_pose = torch.cat([global_orient, body_pose], dim=1)

        # Reshape pose to (batch_size, num_joints, 3)
        num_joints = 24
        pose_reshape = full_pose.reshape(batch_size, num_joints, 3)

        # Get joint locations
        J = torch.einsum('bik,ji->bjk', v_shaped, J_regressor)

        # Convert pose to rotation matrices
        # Reshape to (batch_size * num_joints, 3) for batch processing
        rot_vecs_flat = pose_reshape.reshape(-1, 3)
        rot_mats_flat = batch_rodrigues(rot_vecs_flat)
        rot_mats = rot_mats_flat.view(batch_size, num_joints, 3, 3)

        # Forward kinematics
        pose_feature = rot_mats[:, 1:, :, :] - torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
        pose_feature = pose_feature.view(batch_size, -1)

        # Apply Linear Blend Skinning
        T = self.forward_kinematics(rot_mats, J, parents)

        # Homogeneous coordinates
        v_shaped_homo = F.pad(v_shaped, (0, 1), value=1.0)

        # Apply blend weights
        # weights is (num_verts, num_joints), T is (batch_size, num_joints, 4, 4)
        # We need to compute weighted sum of transformations for each vertex
        T_weighted = torch.einsum('vj,bjxy->bvxy', weights, T)
        v_homo = torch.einsum('bvxy,bvy->bvx', T_weighted, v_shaped_homo)

        vertices = v_homo[:, :, :3]

        # Apply translation
        if transl is not None:
            vertices = vertices + transl.unsqueeze(1)

        return type('Output', (), {'vertices': vertices})()

    def forward_kinematics(self, rot_mats, joints, parents):
        """
        Perform forward kinematics to get transformation matrices.

        Args:
            rot_mats: Rotation matrices (batch_size, num_joints, 3, 3)
            joints: Joint positions (batch_size, num_joints, 3)
            parents: Parent indices for each joint

        Returns:
            Transformation matrices (batch_size, num_joints, 4, 4)
        """
        batch_size = rot_mats.shape[0]
        num_joints = rot_mats.shape[1]
        device = rot_mats.device

        # Initialize transformation matrices
        transforms = torch.zeros((batch_size, num_joints, 4, 4), device=device)

        # Root joint
        transforms[:, 0, :3, :3] = rot_mats[:, 0]
        transforms[:, 0, :3, 3] = joints[:, 0]
        transforms[:, 0, 3, 3] = 1.0

        # Forward kinematics for other joints
        for i in range(1, num_joints):
            parent_idx = parents[i]

            # Local transformation
            local_transform = torch.zeros((batch_size, 4, 4), device=device)
            local_transform[:, :3, :3] = rot_mats[:, i]
            local_transform[:, :3, 3] = joints[:, i] - joints[:, parent_idx]
            local_transform[:, 3, 3] = 1.0

            # Global transformation
            transforms[:, i] = torch.matmul(transforms[:, parent_idx], local_transform)

        # Subtract rest pose
        rest_pose = torch.zeros((batch_size, num_joints, 4, 4), device=device)
        rest_pose[:, :, :3, :3] = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
        rest_pose[:, :, :3, 3] = -joints
        rest_pose[:, :, 3, 3] = 1.0

        transforms = torch.matmul(transforms, rest_pose)

        return transforms

    def __call__(self, **kwargs):
        """Allow calling the model like a function."""
        return self.forward(**kwargs)
