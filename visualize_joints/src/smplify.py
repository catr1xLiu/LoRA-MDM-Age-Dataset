import torch
import os, sys
import pickle
import smplx
import numpy as np
from typing import Union

sys.path.append(os.path.dirname(__file__))
from customloss import (
    camera_fitting_loss,
    body_fitting_loss,
    camera_fitting_loss_3d,
    body_fitting_loss_3d,
)
from prior import MaxMixturePrior
import config


@torch.no_grad()
def guess_init_3d(model_joints, j3d, joints_category="orig"):
    gt_joints = ["RHip", "LHip", "RShoulder", "LShoulder"]
    gt_joints_ind = [config.JOINT_MAP[joint] for joint in gt_joints]

    if joints_category == "orig":
        joints_ind_category = [config.JOINT_MAP[joint] for joint in gt_joints]
    elif joints_category == "AMASS":
        joints_ind_category = [config.AMASS_JOINT_MAP[joint] for joint in gt_joints]
    else:
        print("NO SUCH JOINTS CATEGORY!")

    sum_init_t = (j3d[:, joints_ind_category] - model_joints[:, gt_joints_ind]).sum(
        dim=1
    )
    init_t = sum_init_t / 4.0
    return init_t


class SMPLify3D:
    def __init__(
        self,
        smplxmodel,
        step_size=1e-2,
        batch_size=1,
        num_iters=100,
        use_collision=False,
        use_lbfgs=True,
        joints_category="orig",
        device=torch.device("cuda:0"),
    ):
        self.batch_size = batch_size
        self.device = device
        self.step_size = step_size
        self.num_iters = num_iters
        self.use_lbfgs = use_lbfgs
        self.pose_prior = MaxMixturePrior(
            prior_folder=config.GMM_MODEL_DIR, num_gaussians=8, dtype=torch.float32
        ).to(device)
        self.use_collision = use_collision
        if self.use_collision:
            self.part_segm_fn = config.Part_Seg_DIR
        self.smpl = smplxmodel
        self.model_faces = smplxmodel.faces_tensor.view(-1)
        self.joints_category = joints_category

        if joints_category == "orig":
            self.smpl_index = config.full_smpl_idx
            self.corr_index = config.full_smpl_idx
        elif joints_category == "AMASS":
            self.smpl_index = config.amass_smpl_idx
            self.corr_index = config.amass_idx
        else:
            self.smpl_index = None
            self.corr_index = None
            print("NO SUCH JOINTS CATEGORY!")

    def _pad_to_batch(self, tensor, target_bs):
        actual_bs = tensor.shape[0]
        if actual_bs == target_bs:
            return tensor
        n_extra = target_bs - actual_bs
        pad = tensor[-1:].repeat(n_extra, *[1] * (tensor.ndim - 1))
        return torch.cat([tensor, pad], dim=0)

    def _slice_from_batch(self, tensor, actual_bs):
        if tensor.shape[0] == actual_bs:
            return tensor
        return tensor[:actual_bs]

    def __call__(
        self,
        init_pose,
        init_betas,
        init_cam_t,
        j3d,
        conf_3d: Union[float, torch.Tensor] = 1.0,
        seq_ind=0,
        num_iters_override=None,
        preserve_pose_override=None,
        pose_preserve_weight=0.0,
        skip_stage1=False,
    ):

        actual_bs = init_pose.shape[0]
        model_bs = self.batch_size

        init_pose_pad = self._pad_to_batch(init_pose, model_bs)
        init_betas_pad = self._pad_to_batch(init_betas, model_bs)
        init_cam_t_pad = self._pad_to_batch(init_cam_t, model_bs)
        j3d_pad = self._pad_to_batch(j3d, model_bs)

        if preserve_pose_override is not None:
            preserve_pose_override = self._pad_to_batch(
                preserve_pose_override, model_bs
            )

        conf_3d_tensor = (
            conf_3d
            if isinstance(conf_3d, torch.Tensor)
            else torch.tensor(conf_3d, device=self.device)
        )

        search_tree = None
        pen_distance = None
        filter_faces = None

        if self.use_collision:
            from mesh_intersection.bvh_search_tree import BVH
            import mesh_intersection.loss as collisions_loss
            from mesh_intersection.filter_faces import FilterFaces

            search_tree = BVH(max_collisions=8)
            pen_distance = collisions_loss.DistanceFieldPenetrationLoss(
                sigma=0.5, point2plane=False, vectorized=True, penalize_outside=True
            )
            if self.part_segm_fn:
                part_segm_fn = os.path.expandvars(self.part_segm_fn)
                with open(part_segm_fn, "rb") as faces_parents_file:
                    face_segm_data = pickle.load(faces_parents_file, encoding="latin1")
                faces_segm = face_segm_data["segm"]
                faces_parents = face_segm_data["parents"]
                filter_faces = FilterFaces(
                    faces_segm=faces_segm,
                    faces_parents=faces_parents,
                    ign_part_pairs=None,
                ).to(device=self.device)

        body_pose = init_pose_pad[:, 3:].detach().clone()
        global_orient = init_pose_pad[:, :3].detach().clone()
        betas = init_betas_pad.detach().clone()

        preserve_pose = (
            preserve_pose_override.detach().clone()
            if preserve_pose_override is not None
            else init_pose_pad[:, 3:].detach().clone()
        )

        if not skip_stage1:
            smpl_output = self.smpl(
                global_orient=global_orient, body_pose=body_pose, betas=betas
            )
            model_joints = smpl_output.joints
            init_cam_t = guess_init_3d(
                model_joints, j3d_pad, self.joints_category
            ).detach()
            camera_translation = init_cam_t.clone()

            body_pose.requires_grad = False
            betas.requires_grad = False
            global_orient.requires_grad = True
            camera_translation.requires_grad = True

            camera_opt_params = [global_orient, camera_translation]

            if self.use_lbfgs:
                camera_optimizer = torch.optim.LBFGS(
                    camera_opt_params,
                    max_iter=self.num_iters,
                    lr=self.step_size,
                    line_search_fn="strong_wolfe",
                )
                for _ in range(10):

                    def closure():
                        camera_optimizer.zero_grad()
                        smpl_output = self.smpl(
                            global_orient=global_orient,
                            body_pose=body_pose,
                            betas=betas,
                        )
                        model_joints = smpl_output.joints
                        loss = camera_fitting_loss_3d(
                            model_joints,
                            camera_translation,
                            init_cam_t,
                            j3d_pad,
                            self.joints_category,
                        )
                        loss.backward()
                        return loss

                    camera_optimizer.step(closure)
            else:
                camera_optimizer = torch.optim.Adam(
                    camera_opt_params, lr=self.step_size, betas=(0.9, 0.999)
                )
                for _ in range(20):
                    smpl_output = self.smpl(
                        global_orient=global_orient, body_pose=body_pose, betas=betas
                    )
                    model_joints = smpl_output.joints
                    loss = camera_fitting_loss_3d(
                        model_joints,
                        camera_translation,
                        init_cam_t,
                        j3d_pad,
                        self.joints_category,
                    )
                    camera_optimizer.zero_grad()
                    loss.backward()
                    camera_optimizer.step()
        else:
            camera_translation = init_cam_t_pad.detach().clone()

        body_pose.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = True

        if seq_ind == 0:
            betas.requires_grad = True
            body_opt_params = [body_pose, betas, global_orient, camera_translation]
        else:
            betas.requires_grad = False
            body_opt_params = [body_pose, global_orient, camera_translation]

        num_iters = (
            num_iters_override if num_iters_override is not None else self.num_iters
        )

        if self.use_lbfgs:
            body_optimizer = torch.optim.LBFGS(
                body_opt_params,
                max_iter=self.num_iters,
                lr=self.step_size,
                line_search_fn="strong_wolfe",
            )
            for i in range(self.num_iters):

                def closure():
                    body_optimizer.zero_grad()
                    smpl_output = self.smpl(
                        global_orient=global_orient, body_pose=body_pose, betas=betas
                    )
                    model_joints = smpl_output.joints
                    model_vertices = smpl_output.vertices
                    loss = body_fitting_loss_3d(
                        body_pose,
                        preserve_pose,
                        betas,
                        model_joints[:, self.smpl_index],
                        camera_translation,
                        j3d_pad[:, self.corr_index],
                        self.pose_prior,
                        joints3d_conf=conf_3d_tensor,
                        joint_loss_weight=600.0,
                        pose_preserve_weight=5.0,
                        use_collision=self.use_collision,
                        model_vertices=model_vertices,
                        model_faces=self.model_faces,
                        search_tree=search_tree,
                        pen_distance=pen_distance,
                        filter_faces=filter_faces,
                    )
                    loss.backward()
                    return loss

                body_optimizer.step(closure)
        else:
            body_optimizer = torch.optim.Adam(
                body_opt_params, lr=self.step_size, betas=(0.9, 0.999)
            )
            for i in range(num_iters):
                smpl_output = self.smpl(
                    global_orient=global_orient, body_pose=body_pose, betas=betas
                )
                model_joints = smpl_output.joints
                model_vertices = smpl_output.vertices
                loss = body_fitting_loss_3d(
                    body_pose,
                    preserve_pose,
                    betas,
                    model_joints[:, self.smpl_index],
                    camera_translation,
                    j3d_pad[:, self.corr_index],
                    self.pose_prior,
                    joints3d_conf=conf_3d_tensor,
                    joint_loss_weight=600.0,
                    pose_preserve_weight=pose_preserve_weight,
                    use_collision=self.use_collision,
                    model_vertices=model_vertices,
                    model_faces=self.model_faces,
                    search_tree=search_tree,
                    pen_distance=pen_distance,
                    filter_faces=filter_faces,
                )
                body_optimizer.zero_grad()
                loss.backward()
                body_optimizer.step()

        with torch.no_grad():
            smpl_output = self.smpl(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=betas,
                return_full_pose=True,
            )
            model_joints = smpl_output.joints
            model_vertices = smpl_output.vertices
            final_loss = body_fitting_loss_3d(
                body_pose,
                preserve_pose,
                betas,
                model_joints[:, self.smpl_index],
                camera_translation,
                j3d_pad[:, self.corr_index],
                self.pose_prior,
                joints3d_conf=conf_3d_tensor,
                joint_loss_weight=600.0,
                use_collision=self.use_collision,
                model_vertices=model_vertices,
                model_faces=self.model_faces,
                search_tree=search_tree,
                pen_distance=pen_distance,
                filter_faces=filter_faces,
            )

        vertices = self._slice_from_batch(smpl_output.vertices.detach(), actual_bs)
        joints = self._slice_from_batch(smpl_output.joints.detach(), actual_bs)
        pose = self._slice_from_batch(
            torch.cat([global_orient, body_pose], dim=-1).detach(), actual_bs
        )
        betas = self._slice_from_batch(betas.detach(), actual_bs)
        camera_translation = self._slice_from_batch(
            camera_translation.detach(), actual_bs
        )

        return vertices, joints, pose, betas, camera_translation, final_loss
