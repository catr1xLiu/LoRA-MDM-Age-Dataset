from __future__ import print_function, division
import argparse
import torch
import os, sys, glob
import numpy as np
import joblib
import smplx
import trimesh
import h5py
import multiprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.smplify import SMPLify3D
from src.config import *


def _to_time_joint_xyz(motion, num_joints):
    motion = np.asarray(motion)

    if motion.ndim == 3:
        if motion.shape[1:] == (num_joints, 3):
            return motion.astype(np.float32)
        if motion.shape[:2] == (num_joints, 3):
            return np.transpose(motion, (2, 0, 1)).astype(np.float32)
        raise ValueError(f"Unsupported 3D motion shape: {motion.shape}")

    if motion.ndim == 4:
        if motion.shape[1:3] == (num_joints, 3):
            return np.transpose(motion[0], (2, 0, 1)).astype(np.float32)
        if motion.shape[2:] == (num_joints, 3):
            return motion[0].astype(np.float32)
        raise ValueError(f"Unsupported 4D motion shape: {motion.shape}")

    raise ValueError(f"Unsupported motion rank: {motion.ndim} for shape {motion.shape}")


def peek_motion_file(file_path, num_joints=22):
    loaded = np.load(file_path, allow_pickle=True)

    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "joints" in loaded.files:
            shape = loaded["joints"].shape
            T = shape[0] if shape[0] > shape[-1] else shape[-1]
            return (1, [T])
        if "motion" in loaded.files:
            shape = loaded["motion"].shape
            if len(shape) == 4:
                return (shape[0], [shape[3]] * shape[0])
            T = shape[0]
            return (1, [T])
        raise ValueError(f"NPZ missing supported keys. Found: {list(loaded.files)}")

    elif isinstance(loaded, np.ndarray):
        if loaded.dtype == object:
            payload = loaded.item() if loaded.shape == () else loaded.flat[0]
            if not isinstance(payload, dict) or "motion" not in payload:
                raise ValueError(f"Pickle-npy must be dict with motion key")
            motion = payload["motion"]
            lengths = payload.get("lengths")
            if motion.ndim == 4:
                N = motion.shape[0]
                if lengths is not None:
                    lengths_arr = np.asarray(lengths).reshape(-1)
                    return (N, [int(l) for l in lengths_arr[:N]])
                T_max = motion.shape[3]
                return (N, [T_max] * N)
            else:
                return (1, [motion.shape[0]])
        else:
            return (1, [loaded.shape[0]])
    else:
        raise ValueError(f"Unsupported file type: {type(loaded)}")


def load_motion_file_for_clip(full_path, num_joints, clip_idx=0):
    loaded = np.load(full_path, allow_pickle=True)

    data = None
    pelvis_traj = None
    fps = 20.0
    source_fmt = None

    if isinstance(loaded, np.lib.npyio.NpzFile):
        source_fmt = "npz"
        if "joints" in loaded.files:
            data = np.asarray(loaded["joints"], dtype=np.float32)
            pelvis_traj = (
                np.asarray(loaded["pelvis_traj"], dtype=np.float32)
                if "pelvis_traj" in loaded.files
                else None
            )
            if "fps" in loaded.files:
                fps = float(loaded["fps"])
        elif "motion" in loaded.files:
            data = _to_time_joint_xyz(loaded["motion"], num_joints)
            if "fps" in loaded.files:
                fps = float(loaded["fps"])
        else:
            raise ValueError(f"NPZ missing supported keys. Found: {list(loaded.files)}")
        loaded.close()

    elif isinstance(loaded, np.ndarray):
        if loaded.dtype == object:
            source_fmt = "pickle-npy"
            payload = loaded.item() if loaded.shape == () else loaded.flat[0]
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Pickle npy payload must be dict, got {type(payload)}"
                )
            if "motion" not in payload:
                raise ValueError(
                    f"Pickle dict missing motion. Found keys: {list(payload.keys())}"
                )

            motion_arr = payload["motion"]
            if motion_arr.ndim == 4:
                motion_arr = motion_arr[clip_idx]
            data = _to_time_joint_xyz(motion_arr, num_joints)

            lengths = payload.get("lengths")
            if lengths is not None:
                lengths_arr = np.asarray(lengths).reshape(-1)
                seq_len = (
                    int(lengths_arr[clip_idx])
                    if clip_idx < len(lengths_arr)
                    else lengths_arr[0]
                )
                data = data[:seq_len]

            if "pelvis_traj" in payload:
                pelvis_traj_raw = np.asarray(payload["pelvis_traj"], dtype=np.float32)
                if pelvis_traj_raw.ndim == 3:
                    pelvis_traj = pelvis_traj_raw[clip_idx]
                elif pelvis_traj_raw.ndim == 2:
                    pelvis_traj = pelvis_traj_raw

            if "fps" in payload:
                fps = float(payload["fps"])

        else:
            source_fmt = "plain-npy"
            data = _to_time_joint_xyz(loaded, num_joints)

    else:
        raise ValueError(f"Unsupported loaded file type: {type(loaded)}")

    if pelvis_traj is None:
        pelvis_traj = np.zeros((data.shape[0], 3), dtype=np.float32)

    pelvis_traj = np.asarray(pelvis_traj, dtype=np.float32)
    if pelvis_traj.shape[0] < data.shape[0]:
        pad = np.repeat(pelvis_traj[-1:], data.shape[0] - pelvis_traj.shape[0], axis=0)
        pelvis_traj = np.concatenate([pelvis_traj, pad], axis=0)
    elif pelvis_traj.shape[0] > data.shape[0]:
        pelvis_traj = pelvis_traj[: data.shape[0]]

    if pelvis_traj.ndim != 2 or pelvis_traj.shape[1] != 3:
        raise ValueError(f"Invalid pelvis_traj shape: {pelvis_traj.shape}")

    return data, pelvis_traj, fps, source_fmt


def is_clip_done(save_folder, global_clip_idx, expected_frames):
    clip_dir = os.path.join(save_folder, f"clip_{global_clip_idx:04d}")
    if not os.path.isdir(clip_dir):
        return False
    pkl_files = glob.glob(os.path.join(clip_dir, "*.pkl"))
    return len(pkl_files) >= expected_frames


def saver_fn(save_queue, save_folder):
    while True:
        item = save_queue.get()
        if item is None:
            break

        global_clip_idx, clip_results, expected_frames, smpl_faces = item

        dir_save = os.path.join(save_folder, f"clip_{global_clip_idx:04d}")
        os.makedirs(dir_save, exist_ok=True)

        for idx, frame_data in enumerate(clip_results):
            out_verts = frame_data["vertices"]
            mesh = trimesh.Trimesh(vertices=out_verts, faces=smpl_faces, process=False)
            mesh.export(os.path.join(dir_save, f"{idx:04d}.ply"))

            param = {
                "beta": frame_data["betas"],
                "pose": frame_data["pose"],
                "cam": frame_data["cam_t"],
                "root": frame_data["cam_t"].squeeze(),
            }
            joblib.dump(param, os.path.join(dir_save, f"{idx:04d}.pkl"), compress=3)


def worker_fn(gpu_id, work_queue, save_queue, args):
    try:
        device = torch.device(f"cuda:{gpu_id}")
        batch_size = args.batch_size
        print(
            f"[Worker GPU {gpu_id}] Initializing on {device}, batch_size={batch_size}"
        )

        smplmodel = smplx.create(
            SMPL_MODEL_DIR,
            model_type="smpl",
            gender="neutral",
            ext="pkl",
            batch_size=batch_size,
        ).to(device)
        smpl_faces = smplmodel.faces.copy()

        with h5py.File(SMPL_MEAN_FILE, "r") as f:
            init_mean_pose = (
                torch.from_numpy(f["pose"][:]).unsqueeze(0).float().to(device)
            )
            init_mean_shape = (
                torch.from_numpy(f["shape"][:]).unsqueeze(0).float().to(device)
            )

        smplify = SMPLify3D(
            smplxmodel=smplmodel,
            batch_size=batch_size,
            joints_category=args.joint_category,
            num_iters=args.num_smplify_iters,
            use_lbfgs=False,
            device=device,
        )

        confidence_input = torch.ones(args.num_joints)
        if args.fix_foot == "True":
            confidence_input[7] = 1.5
            confidence_input[8] = 1.5
            confidence_input[10] = 1.5
            confidence_input[11] = 1.5
        conf_batch = confidence_input.to(device)

        print(f"[Worker GPU {gpu_id}] Ready")

        while True:
            item = work_queue.get()
            if item is None:
                break

            file_path, clip_idx_within_file, global_clip_idx, expected_frames = item

            print(
                f"[Worker GPU {gpu_id}] Processing clip {global_clip_idx:04d} from {os.path.basename(file_path)} (clip {clip_idx_within_file})"
            )

            try:
                data, pelvis_traj, fps, source_fmt = load_motion_file_for_clip(
                    file_path, args.num_joints, clip_idx=clip_idx_within_file
                )

                if data.shape[0] != expected_frames:
                    print(
                        f"[Worker GPU {gpu_id}] Warning: expected {expected_frames} frames, got {data.shape[0]}"
                    )

                clip_results = []
                prev_last_pose = None
                prev_last_betas = None

                for batch_start in range(0, data.shape[0], batch_size):
                    batch_end = min(batch_start + batch_size, data.shape[0])
                    actual_bs = batch_end - batch_start

                    j3d_batch = torch.stack(
                        [
                            torch.tensor(data[i] + pelvis_traj[i], dtype=torch.float32)
                            for i in range(batch_start, batch_end)
                        ]
                    ).to(device)

                    if batch_start == 0:
                        init_pose_batch = (
                            init_mean_pose.expand(actual_bs, -1).clone().to(device)
                        )
                        init_betas_batch = (
                            init_mean_shape.expand(actual_bs, -1).clone().to(device)
                        )
                    else:
                        init_pose_batch = prev_last_pose.expand(actual_bs, -1).clone()
                        init_betas_batch = prev_last_betas.expand(actual_bs, -1).clone()

                    init_cam_t_batch = torch.tensor(
                        pelvis_traj[batch_start:batch_end], dtype=torch.float32
                    ).to(device)

                    new_vertices, _, new_pose, new_betas, new_cam_t, _ = smplify(
                        init_pose_batch,
                        init_betas_batch,
                        init_cam_t_batch,
                        j3d_batch,
                        conf_3d=conf_batch,
                        seq_ind=0 if batch_start == 0 else 1,
                    )

                    prev_last_pose = new_pose[-1:].detach()
                    prev_last_betas = new_betas[-1:].detach()

                    for i in range(actual_bs):
                        frame_idx = batch_start + i
                        with torch.no_grad():
                            out = smplmodel(
                                betas=new_betas[i : i + 1],
                                global_orient=new_pose[i : i + 1, :3],
                                body_pose=new_pose[i : i + 1, 3:],
                                transl=new_cam_t[i : i + 1],
                                return_verts=True,
                            )

                        clip_results.append(
                            {
                                "pose": new_pose[i : i + 1].detach().cpu().numpy(),
                                "betas": new_betas[i : i + 1].detach().cpu().numpy(),
                                "cam_t": new_cam_t[i : i + 1].detach().cpu().numpy(),
                                "vertices": out.vertices.detach()
                                .cpu()
                                .numpy()
                                .squeeze(),
                            }
                        )

                save_queue.put(
                    (global_clip_idx, clip_results, expected_frames, smpl_faces)
                )
                print(
                    f"[Worker GPU {gpu_id}] Done clip {global_clip_idx:04d} ({data.shape[0]} frames)"
                )

            except Exception as e:
                import traceback

                print(
                    f"[Worker GPU {gpu_id}] Error processing clip {global_clip_idx:04d}: {e}"
                )
                traceback.print_exc()
                continue

    except Exception as e:
        import traceback

        print(f"[Worker GPU {gpu_id}] Fatal error: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=1, help="number of GPUs to use")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size per GPU (frames processed in parallel)",
    )
    parser.add_argument(
        "--num_smplify_iters", type=int, default=150, help="num of smplify iters"
    )
    parser.add_argument("--num_joints", type=int, default=22, help="joint number")
    parser.add_argument(
        "--joint_category",
        type=str,
        default="AMASS",
        help="joint correspondence category",
    )
    parser.add_argument("--fix_foot", type=str, default="False", help="fix foot or not")
    parser.add_argument(
        "--data_folder", type=str, required=True, help="folder containing motion files"
    )
    parser.add_argument(
        "--save_folder", type=str, required=True, help="results save folder"
    )
    args = parser.parse_args()

    print(f"=== Multi-GPU SMPL Fitting ===")
    print(f"Data folder: {args.data_folder}")
    print(f"Save folder: {args.save_folder}")
    print(f"Num GPUs: {args.num_gpus}")
    print(f"Batch size: {args.batch_size}")
    print(f"SMPLify iterations: {args.num_smplify_iters}")
    print(f"Joint category: {args.joint_category}")
    print(f"Fix foot: {args.fix_foot}")
    print()

    os.makedirs(args.save_folder, exist_ok=True)

    npy_files = sorted(
        glob.glob(os.path.join(args.data_folder, "**/*.npy"), recursive=True)
    )
    npz_files = sorted(
        glob.glob(os.path.join(args.data_folder, "**/*.npz"), recursive=True)
    )
    all_files = npy_files + npz_files

    if not all_files:
        print("No .npy or .npz files found.")
        return

    print(f"Found {len(all_files)} files")

    work_items = []
    global_idx = 0

    for file_path in all_files:
        try:
            n_clips, frame_counts = peek_motion_file(file_path)
            print(f"{os.path.basename(file_path)}: {n_clips} clip(s) -> {frame_counts}")
            for clip_idx in range(n_clips):
                expected_frames = frame_counts[clip_idx]
                if not is_clip_done(args.save_folder, global_idx, expected_frames):
                    work_items.append(
                        (file_path, clip_idx, global_idx, expected_frames)
                    )
                else:
                    print(f"  Clip {global_idx:04d} already done, skipping")
                global_idx += 1
        except Exception as e:
            print(f"Error peeking {file_path}: {e}")
            continue

    if not work_items:
        print("All clips already processed.")
        return

    print(f"\nTotal work items: {len(work_items)}")

    ctx = multiprocessing.get_context("spawn")

    work_queue = ctx.Queue()
    save_queue = ctx.Queue(maxsize=args.num_gpus * 2)

    for item in work_items:
        work_queue.put(item)

    for _ in range(args.num_gpus):
        work_queue.put(None)

    print("Starting saver process...")
    saver_proc = ctx.Process(target=saver_fn, args=(save_queue, args.save_folder))
    saver_proc.start()

    print(f"Starting {args.num_gpus} worker process(es)...")
    workers = []
    for gpu_id in range(args.num_gpus):
        p = ctx.Process(target=worker_fn, args=(gpu_id, work_queue, save_queue, args))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()
        print(f"Worker {p.pid} finished")

    print("All workers done, signaling saver...")
    save_queue.put(None)
    saver_proc.join()
    print("=== Done ===")


if __name__ == "__main__":
    main()
