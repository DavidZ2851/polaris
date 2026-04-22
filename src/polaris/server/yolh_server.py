"""Policy inference server for YOLH point-cloud policy.

Usage:
    python src/polaris/server/yolh_server.py \
        --ckpt_path /path/to/checkpoint.pt \
        --config_path /path/to/deployment.yaml \
        --dataset_meta /path/to/dataset_meta.npz \
        --port 5557
"""

import argparse
import pickle
from typing import Optional

import MinkowskiEngine as ME
import numpy as np
import torch
import zmq

from polaris.policy.yolh import YOLH
from polaris.policy.yolh.utils.config import load_action_norm_stats, load_config
from polaris.policy.yolh.utils.constants import IMG_MEAN, IMG_STD
from polaris.policy.yolh.utils.observation import build_observation_cloud
from polaris.policy.yolh.utils.transformation import project_action_to_base, unnormalize_action


def _cloud_to_sparse(cloud: np.ndarray, voxel_size: float, device: torch.device):
    coords = np.ascontiguousarray(cloud[:, :3] / voxel_size, dtype=np.int32)
    feats = np.ascontiguousarray(cloud.astype(np.float32))
    coords_batch, feats_batch = ME.utils.sparse_collate([coords], [feats])
    return ME.SparseTensor(feats_batch.to(device), coords_batch.to(device))


def _normalize_intrinsic(intrinsic: np.ndarray) -> np.ndarray:
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if intrinsic.shape == (3, 3):
        return np.array(
            [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]],
            dtype=np.float32,
        )
    if intrinsic.shape == (4,):
        return intrinsic
    raise ValueError(
        f"Unsupported camera_intrinsic shape {intrinsic.shape}; expected (3, 3) or (4,)"
    )


def _load_runtime_cfg(config_path: str, dataset_meta: Optional[str]) -> dict:
    cfg = load_config(config_path)
    cfg["camera_intrinsic"] = _normalize_intrinsic(cfg["camera_intrinsic"])

    meta_path = dataset_meta or cfg.get("dataset_meta")
    if meta_path is None:
        raise ValueError("Provide --dataset_meta or set dataset_meta in config")

    stats = load_action_norm_stats(str(meta_path))
    cfg.update({key: value for key, value in stats.items() if value is not None})

    for key in ("safe_workspace_min", "safe_workspace_max"):
        if key in cfg and cfg[key] is not None:
            cfg[key] = np.asarray(cfg[key], dtype=np.float32)

    return cfg


def _load_policy(ckpt_path: str, cfg: dict, device: torch.device) -> YOLH:
    model_cfg = cfg["model"]
    policy = YOLH(
        num_action=cfg["num_action"],
        input_dim=6,
        obs_feature_dim=model_cfg["obs_feature_dim"],
        action_dim=10,
        hidden_dim=model_cfg["hidden_dim"],
        nheads=model_cfg["nheads"],
        num_encoder_layers=model_cfg["num_encoder_layers"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        dropout=model_cfg["dropout"],
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy.load_state_dict(state, strict=False)
    policy.to(device)
    policy.eval()
    return policy


def _postprocess_action_chunk(action_chunk: np.ndarray, cfg: dict) -> np.ndarray:
    action_chunk = unnormalize_action(action_chunk, cfg)
    action_chunk = project_action_to_base(action_chunk, cfg["cam_to_base"])

    safe_min = cfg.get("safe_workspace_min")
    safe_max = cfg.get("safe_workspace_max")
    if safe_min is not None and safe_max is not None:
        safe_eps = float(cfg.get("safe_eps", 0.002))
        action_chunk[..., :3] = np.clip(
            action_chunk[..., :3],
            safe_min + safe_eps,
            safe_max - safe_eps,
        )

    close_threshold = float(
        cfg.get("gripper_close_threshold", float(cfg["max_gripper_width"]) * 0.5)
    )
    action_chunk[..., -1] = (action_chunk[..., -1] <= close_threshold).astype(np.float32)

    return action_chunk.astype(np.float32)


def _gripper_points_to_cloud(gripper_pcd: np.ndarray, cfg: dict) -> np.ndarray:
    if len(gripper_pcd) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    cam_to_base = np.asarray(cfg["cam_to_base"], dtype=np.float32)
    base_to_cam = np.linalg.inv(cam_to_base)
    rot = base_to_cam[:3, :3]
    trans = base_to_cam[:3, 3]

    gripper_cam = (rot @ gripper_pcd.T).T + trans
    gripper_color = (np.array([1.0, 0.55, 0.0], dtype=np.float32) - IMG_MEAN) / IMG_STD
    gripper_colors = np.repeat(gripper_color[None, :], len(gripper_cam), axis=0)
    return np.concatenate([gripper_cam, gripper_colors], axis=-1).astype(np.float32)


def _request_to_point_cloud(request: dict, cfg: dict) -> np.ndarray:
    if "point_cloud" in request:
        return np.asarray(request["point_cloud"], dtype=np.float32)

    rgb = np.asarray(request["rgb"])
    depth = np.asarray(request["depth"])
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    robot_mask = request.get("robot_mask")
    if robot_mask is not None:
        robot_mask = np.asarray(robot_mask).astype(bool)
        if robot_mask.shape == depth.shape:
            depth = np.where(robot_mask, 0.0, depth)

    cloud = build_observation_cloud(
        rgb=rgb,
        depth=depth,
        intrinsic=cfg["camera_intrinsic"],
        cfg=cfg,
    )
    gripper_cloud = _gripper_points_to_cloud(
        np.asarray(request["gripper_pcd"], dtype=np.float32),
        cfg,
    )
    if len(gripper_cloud) == 0:
        return cloud.astype(np.float32)
    if len(cloud) == 0:
        return gripper_cloud
    return np.concatenate([cloud, gripper_cloud], axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset_meta", type=str, default=None)
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    print(f"Loading YOLH config from {args.config_path}...")
    cfg = _load_runtime_cfg(args.config_path, args.dataset_meta)
    print(f"Loading YOLH checkpoint from {args.ckpt_path} on {device}...")
    policy = _load_policy(args.ckpt_path, cfg, device)
    print("YOLH policy loaded.")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"YOLH server listening on port {args.port}")

    while True:
        raw = socket.recv()
        request = pickle.loads(raw)

        if request.get("reset"):
            socket.send(pickle.dumps({"status": "ok"}))
            continue

        point_cloud = _request_to_point_cloud(request, cfg)
        if len(point_cloud) == 0:
            socket.send(pickle.dumps({"action_chunk": np.zeros((0, 10), dtype=np.float32)}))
            continue

        with torch.inference_mode():
            sparse_cloud = _cloud_to_sparse(point_cloud, cfg["voxel_size"], device)
            action_chunk = policy(sparse_cloud, actions=None, batch_size=1)

        action_chunk = action_chunk.squeeze(0).detach().cpu().numpy()
        action_chunk = _postprocess_action_chunk(action_chunk, cfg)
        socket.send(pickle.dumps({"action_chunk": action_chunk}))


if __name__ == "__main__":
    main()