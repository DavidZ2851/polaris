"""Configuration loading utilities for inference."""

import numpy as np
import yaml


def load_config(path: str) -> dict:
    """Load camera calibration / deployment YAML and parse arrays."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["cam_to_base"] = np.array(cfg["cam_to_base"], dtype=np.float64).reshape(4, 4)
    cfg["workspace_min"] = np.array(cfg["workspace_min"], dtype=np.float32)
    cfg["workspace_max"] = np.array(cfg["workspace_max"], dtype=np.float32)
    cfg["gripper_offset"] = cfg.get("gripper_offset", [0.05, 0.0, 0.0])
    cfg["camera_intrinsic"] = np.array(cfg["camera_intrinsic"], dtype=np.float32)
    return cfg


def load_action_norm_stats(dataset_path: str) -> dict:
    """Load action normalisation statistics from a dataset .npz file."""
    data = np.load(dataset_path, allow_pickle=True)
    required = ["trans_min", "trans_max", "max_gripper_width"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            f"Dataset meta missing keys {missing}: {dataset_path}. "
            "Expected output from scripts/merge_episodes.py"
        )
    max_w = float(data["max_gripper_width"])
    if max_w <= 0:
        raise ValueError(
            f"Invalid max_gripper_width={max_w} from dataset meta: {dataset_path}"
        )
    return {
        "trans_min": np.asarray(data["trans_min"], dtype=np.float32),
        "trans_max": np.asarray(data["trans_max"], dtype=np.float32),
        "max_gripper_width": np.float32(max_w),
        "num_action": int(data["num_action"]) if "num_action" in data else None,
        "task_name": str(data["task_name"]) if "task_name" in data else None,
    }
