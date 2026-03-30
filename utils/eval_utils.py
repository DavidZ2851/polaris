import numpy as np
from scipy.spatial.transform import Rotation
import torch

def randomize_object_poses(randomization_setting: dict, initial_conditions: list) -> list:
    result = dict(initial_conditions[0])

    for obj, ranges in randomization_setting.items():
        x = np.random.uniform(*ranges["x"])
        y = np.random.uniform(*ranges["y"])
        z = np.random.uniform(*ranges["z"])

        rx = np.random.uniform(*ranges.get("ori_x", (0, 0)))
        ry = np.random.uniform(*ranges.get("ori_y", (0, 0)))
        rz = np.random.uniform(*ranges.get("ori_z", (0, 0)))

        ori_pose = initial_conditions[0][obj]  # [x, y, z, qx, qy, qz, qw] xyzw
        orig_q = Rotation.from_quat(ori_pose[3:7])  # already xyzw
        rel_q = Rotation.from_euler("xyz", [rx, ry, rz])
        final_q = orig_q * rel_q
        q = final_q.as_quat()  # xyzw
        result[obj] = [x, y, z, q[3], q[0], q[1], q[2]]  # wxyz for Isaac Sim

        print(f"[{obj}] pos=({x:.3f}, {y:.3f}, {z:.3f})  quat(wxyz)=({q[3]:.3f}, {q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f})")
    return [result]

def is_success(info) -> bool:
    try:
        val = info["rubric"]["success"]
        v = val[0] if hasattr(val, "__len__") else val
        if isinstance(v, torch.Tensor):
            return bool(v.item())
        return bool(v)
    except (KeyError, TypeError, IndexError) as e:
        print(f"  [warn] Could not read info['rubric']['success']: {e}")
        return False