import torch
import argparse
import gymnasium as gym
from isaaclab.app import AppLauncher
import imageio.v3 as iio
import numpy as np
import json
import h5py
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output videos")
parser.add_argument("--demo_file", type=str, required=True, help="Path to input .hdf5 demo file")
parser.add_argument("--demo_num", type=int, default=None, help="Demo number to replay (e.g. 91 for demo_91). If omitted, all demos are replayed.")
parser.add_argument("--max_steps", type=int, default=None, help="Max steps to replay (default: all)")
parser.add_argument("--max_dpos", type=float, default=0.05, help="Position delta normalization scale")
parser.add_argument("--max_drot", type=float, default=0.5, help="Rotation delta normalization scale")
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import polaris.environments
from isaaclab_tasks.utils import parse_env_cfg
from polaris.environments.manager_based_rl_splat_environment import ManagerBasedRLSplatEnv
from polaris.utils import load_eval_initial_conditions, DATA_PATH
from polaris.utils_.planner_utils import setup_curobo
from curobo.types.math import Pose

DEVICE = "cuda:0"
CONTROLLER = "DROID-PutRedCup-no-curtain"

env_cfg = parse_env_cfg(
    CONTROLLER,
    device="cuda",
    num_envs=1,
    use_fabric=True,
)

env: ManagerBasedRLSplatEnv = gym.make(CONTROLLER, cfg=env_cfg)

language_instruction, initial_conditions = load_eval_initial_conditions(env.usd_file)

obs, info = env.reset(object_positions=initial_conditions[0], expensive=True)

motion_gen = setup_curobo()


def delta_to_absolute(
    action_delta: np.ndarray,
    curr_pos: np.ndarray,
    curr_quat_wxyz: np.ndarray,
    max_dpos: float = 0.05,
    max_drot: float = 0.5,
) -> torch.Tensor:
    """
    Convert a normalized 7-DOF delta action to an absolute EE pose.

    Args:
        action_delta:    (7,) = delta_pos(3) + delta_axis_angle(3) + gripper(1), normalized
        curr_pos:        (3,) current EE position
        curr_quat_wxyz:  (4,) current EE quaternion in wxyz convention
        max_dpos:        normalization scale for position delta
        max_drot:        normalization scale for rotation delta

    Returns:
        (8,) tensor = pos(3) + quat_wxyz(4) + gripper(1), gripper in [0, 1]
    """
    delta_pos = action_delta[:3] * max_dpos
    delta_aa  = action_delta[3:6] * max_drot
    gripper   = action_delta[6]

    target_pos = curr_pos + delta_pos

    # wxyz -> xyzw for scipy
    curr_quat_xyzw = curr_quat_wxyz[[1, 2, 3, 0]]
    curr_rot = R.from_quat(curr_quat_xyzw)

    delta_rot  = R.from_rotvec(delta_aa)
    target_rot = delta_rot * curr_rot

    xyzw = target_rot.as_quat()
    target_quat_wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

    # [-1, +1] -> [0, 1]
    gripper_abs = (gripper + 1.0) / 2.0

    result = np.concatenate([target_pos, target_quat_wxyz, [gripper_abs]]).astype(np.float32)
    return torch.from_numpy(result).to(DEVICE)


def ee_to_joint(action_ee: torch.Tensor) -> torch.Tensor:
    """
    Convert absolute EE pose (8,) to joint solution (1, 8) via IK.
    """
    ee_pos    = action_ee[:3].to(DEVICE)
    ee_quat   = action_ee[3:7].to(DEVICE)   # wxyz
    gripper   = action_ee[-1]

    goal_pose  = Pose(position=ee_pos, quaternion=ee_quat)
    ik_result  = motion_gen.ik_solver.solve_single(goal_pose)
    joint_solution = ik_result.solution.squeeze(0)[:, :8]
    joint_solution[:, -1] = gripper

    return joint_solution  # (1, 8)


# ── Determine which demos to replay ──────────────────────────────────────────

from pathlib import Path
output_dir = Path(args_cli.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

with h5py.File(args_cli.demo_file, "r") as f:
    all_demo_keys = sorted(f["data"].keys())  # e.g. ['demo_0', 'demo_1', ...]

if args_cli.demo_num is not None:
    demo_keys = [f"demo_{args_cli.demo_num}"]
    missing = [k for k in demo_keys if k not in all_demo_keys]
    if missing:
        raise ValueError(f"{missing} not found. Available: {all_demo_keys}")
else:
    demo_keys = all_demo_keys

print(f"Will replay {len(demo_keys)} demo(s): {demo_keys}")

# ── Per-demo replay loop ──────────────────────────────────────────────────────

for demo_key in demo_keys:
    print(f"\n{'='*60}")
    print(f"Loading {demo_key} from {args_cli.demo_file}")

    with h5py.File(args_cli.demo_file, "r") as f:
        grp        = f[f"data/{demo_key}"]
        actions    = grp["actions"][:]              # (T, 7)

        eef_pos    = grp["obs/robot0_eef_pos"][:]   # (T, 3)
        eef_quat   = grp["obs/robot0_eef_quat"][:]  # (T, 4) xyzw
        num_samples = grp.attrs["num_samples"]

    print(f"  num_samples : {num_samples}  |  actions: {actions.shape}")

    total_steps = num_samples
    if args_cli.max_steps is not None:
        total_steps = min(total_steps, args_cli.max_steps)
    print(f"  Replaying {total_steps} steps")

    obs, info = env.reset(object_positions=initial_conditions[0], expensive=True)

    frames       = [obs["splat"]["cam0"]]
    frames_2     = [obs["splat"]["cam1"]]
    wrist_frames = [obs["splat"]["wrist_cam"]]

    for i in range(total_steps):
        delta_action   = actions[i]                         # (7,)
        curr_pos       = eef_pos[i]                         # (3,)
        curr_quat_xyzw = eef_quat[i]                        # (4,) xyzw -> wxyz
        curr_quat_wxyz = curr_quat_xyzw[[3, 0, 1, 2]]

        action_ee = delta_to_absolute(
            delta_action,
            curr_pos,
            curr_quat_wxyz,
            max_dpos=args_cli.max_dpos,
            max_drot=args_cli.max_drot,
        )

        joint_cmd = ee_to_joint(action_ee)

        print(f"  Step {i:4d} | pos={action_ee[:3].cpu().numpy()} "
              f"gripper={action_ee[-1].item():.3f}")

        obs, rew, term, trunc, info = env.step(joint_cmd, expensive=True)

        frames.append(obs["splat"]["cam0"])
        frames_2.append(obs["splat"]["cam1"])
        wrist_frames.append(obs["splat"]["wrist_cam"])

        if term or trunc:
            print(f"  Episode ended at step {i} (term={term}, trunc={trunc})")
            break

    # Save videos for this demo
    for name, buf in [
        (f"{demo_key}_cam0.mp4",  frames),
        (f"{demo_key}_cam1.mp4",  frames_2),
        (f"{demo_key}_wrist.mp4", wrist_frames),
    ]:
        arr = np.stack(buf, axis=0)
        iio.imwrite(str(output_dir / name), arr, fps=30)
        print(f"  Saved {output_dir / name}")

env.close()
simulation_app.close()