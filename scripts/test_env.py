import torch
import argparse
import gymnasium as gym
from isaaclab.app import AppLauncher
import imageio.v3 as iio
import numpy as np
import json

parser = argparse.ArgumentParser()
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import polaris.environments
from isaaclab_tasks.utils import parse_env_cfg
from polaris.environments.manager_based_rl_splat_environment import ManagerBasedRLSplatEnv
from polaris.utils import load_eval_initial_conditions
from polaris.utils_.vis_utils import debug_plot
import omni.replicator.core as rep

env_cfg = parse_env_cfg(
    "DROID-PutRedCup-no-curtain-osc",
    device="cuda",
    num_envs=1,
    use_fabric=True,
)


def debug_droid_cfg(env, obs):
    """
    Cross-check all frames: ee_pose obs, base_link body, OSC offset, gripper_pcd.
    Call after env.reset() and after first env.step().
    """
    import torch
    from scipy.spatial.transform import Rotation
    import numpy as np
    from isaaclab.utils.math import combine_frame_transforms

    robot = env.unwrapped.scene["robot"]
    arm_action = env.unwrapped.action_manager._terms["arm"]

    print("=" * 60)

    # ── 1. Raw base_link body pose ────────────────────────────────
    base_link_idx = robot.find_bodies("base_link")[0][0]
    base_pos  = robot.data.body_pos_w[0, base_link_idx]
    base_quat = robot.data.body_quat_w[0, base_link_idx]  # wxyz
    print(f"[1] base_link pos        : {base_pos.cpu().numpy().round(4)}")
    print(f"[1] base_link quat (wxyz): {base_quat.cpu().numpy().round(4)}")

    # ── 2. ee_pose obs (grasp frame = base_link + offset) ─────────
    ee = obs["policy"]["ee_pose"][0]
    print(f"[2] ee_pose obs pos       : {ee[:3].cpu().numpy().round(4)}")
    print(f"[2] ee_pose obs quat(wxyz): {ee[3:7].cpu().numpy().round(4)}")

    # ── 3. Verify offset math manually ───────────────────────────
    from polaris.environments.droid_cfg import _GRASP_OFFSET_POS, _GRASP_OFFSET_QUAT
    device = base_pos.device
    bp = base_pos.unsqueeze(0)
    bq = base_quat.unsqueeze(0)
    op = _GRASP_OFFSET_POS.to(device).unsqueeze(0)
    oq = _GRASP_OFFSET_QUAT.to(device).unsqueeze(0)
    gp, gq = combine_frame_transforms(bp, bq, op, oq)
    print(f"[3] manual grasp pos      : {gp[0].cpu().numpy().round(4)}")
    print(f"[3] manual grasp quat(wxyz): {gq[0].cpu().numpy().round(4)}")
    print(f"[3] matches ee_pose obs?  : pos={torch.allclose(gp[0], ee[:3], atol=1e-4)} quat={torch.allclose(gq[0], ee[3:7], atol=1e-4)}")

    # ── 4. OSC body_offset config ─────────────────────────────────
    print(f"[4] OSC body_name         : {arm_action.cfg.body_name}")
    print(f"[4] OSC offset pos        : {arm_action.cfg.body_offset.pos}")
    print(f"[4] OSC offset rot (wxyz) : {arm_action.cfg.body_offset.rot}")

    # ── 5. What action should we send to hold home? ───────────────
    # OSC expects: [pos(3), quat_xyzw(4), gripper(1)]
    # ee_pose obs gives: [pos(3), quat_wxyz(4), gripper(1)]
    action_wxyz = torch.cat([ee[:3], ee[3:7], ee[7:8]], dim=0)
    action_xyzw = torch.cat([ee[:3], ee[4:7], ee[3:4], ee[7:8]], dim=0)
    print(f"[5] action if wxyz        : {action_wxyz.cpu().numpy().round(4)}")
    print(f"[5] action if xyzw        : {action_xyzw.cpu().numpy().round(4)}")

    # ── 6. Rotation sanity check ──────────────────────────────────
    quat_wxyz = ee[3:7].cpu().numpy()
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
    R = Rotation.from_quat(quat_xyzw)
    print(f"[6] grasp euler (xyz deg) : {R.as_euler('xyz', degrees=True).round(2)}")
    print(f"[6] grasp Z-axis → world  : {R.apply([0,0,1]).round(4)}  (should point down ~[0,0,-1])")
    print(f"[6] grasp X-axis → world  : {R.apply([1,0,0]).round(4)}  (should point forward)")

    # ── 7. gripper_pcd sanity ─────────────────────────────────────
    pcd = obs["policy"]["gripper_pcd"][0].cpu().numpy()  # (4, 3)
    print(f"[7] gripper_pcd[0] top    : {pcd[0].round(4)}")
    print(f"[7] gripper_pcd[3] grasp  : {pcd[3].round(4)}")
    print(f"[7] grasp == ee_pos?      : {np.allclose(pcd[3], ee[:3].cpu().numpy(), atol=1e-3)}")

    # ── 8. OSC action dim ─────────────────────────────────────────
    print(f"[8] OSC action_dim        : {arm_action.action_dim}")
    print(f"[8] finger action_dim     : {env.unwrapped.action_manager._terms['finger_joint'].action_dim}")
    print(f"[8] total action_dim      : {env.unwrapped.action_space.shape}")

    print("=" * 60)


def get_cam_param(calibration_path):
    with open(calibration_path, "r") as f:
        cams = json.load(f)

    calibration = {}
    for name, c in cams.items():
        calibration[name] = {
            "intrinsic":  np.array(c["intrinsic"]),
            "extrinsic":  np.array(c["extrinsic"]),  # cam_to_base (4, 4)
            "distortion": np.array(c["distortion"]),
        }

    return calibration


env: ManagerBasedRLSplatEnv = gym.make("DROID-PutRedCup-no-curtain-osc", cfg=env_cfg)

joint_limits = torch.tensor([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
    [-2.8973,  2.8973],
    [ 0.0000,  0.7854],
    [ 0.0000,  0.7854],
    [-3.1416,  3.1416],
    [-3.1416,  3.1416],
    [-3.1416,  3.1416],
    [-3.1416,  3.1416],
], device="cuda:0", dtype=torch.float32)

def sample_random_action(joint_limits: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
    lower = joint_limits[:, 0]
    upper = joint_limits[:, 1]
    u = torch.rand(
        (batch_size, lower.shape[0]),
        device=joint_limits.device,
        dtype=joint_limits.dtype,
    )
    action = lower.unsqueeze(0) + (upper - lower).unsqueeze(0) * u
    return action[:, :8]  # env action is 8D, first 7 are arm joints


language_instruction, initial_conditions = load_eval_initial_conditions(env.usd_file)
calibration = get_cam_param("/home/haotian/polaris/PolaRiS-Hub/put_red_cup_no_curtain/cam_calibration.json")
# obs, info = env.reset(object_positions=initial_conditions[0], expensive=True)

# robot = env.unwrapped.scene["robot"]
# base_link_idx = robot.find_bodies("base_link")[0][0]

# # What ee_pose obs computes
# print("ee_pose from obs:", obs["policy"]["ee_pose"][0, :3])

# # What base_link raw position is
# base_pos  = robot.data.body_pos_w[0, base_link_idx]
# base_quat = robot.data.body_quat_w[0, base_link_idx]
# print("base_link pos   :", base_pos)
# print("base_link quat  :", base_quat)

# # What OSC thinks its EE is — check via action manager
# arm_action = env.unwrapped.action_manager._terms["arm"]
# print("OSC EE pos      :", arm_action._ee_pose_b)   # in robot base frame
# print("OSC body name   :", arm_action.cfg.body_name)

# robot = env.scene["robot"]

# print("isaac sim ee_pose: ", obs["policy"]["ee_pose"])

# print()

obs, info = env.reset(object_positions=initial_conditions[0], expensive=True)
debug_droid_cfg(env, obs)

from scipy.spatial.transform import Rotation
import numpy as np


arm_action = env.unwrapped.action_manager._terms["arm"]

# OSC tracks these internally after each step
print("EE pos (robot base frame):", arm_action._ee_pose_b)   # (N, 7)


# To get world frame, transform by robot base pose
robot = env.unwrapped.scene["robot"]
root_pos  = robot.data.root_pos_w   # (N, 3)
root_quat = robot.data.root_quat_w  # (N, 4) wxyz

from isaaclab.utils.math import combine_frame_transforms
ee_pos_w, ee_quat_w = combine_frame_transforms(
    root_pos, root_quat,
    arm_action._ee_pose_b[:, :3], arm_action._ee_pose_b[:, 3:7]
)
print("EE pos  (world):", ee_pos_w)   # (N, 3)
print("EE quat (world):", ee_quat_w)  # (N, 4) wxyz


step = 0
max_steps = 50
frames = [obs["splat"]["cam0"]]
frames_2 = [obs["splat"]["cam1"]]
wrist_frames = [obs["splat"]["wrist_cam"]]


import torch

# action_1 = torch.tensor([[0.01968304, -0.4766486 ,  0.00991582, -2.38033551,  0.01029314, 1.95322416, -0.00605614, 0]], device="cuda", dtype=torch.float32)
action_0 = torch.tensor([[0.0000, -0.6283,  0.0000, -2.5133,  0.0000,  1.8850,  0.0000,  0.000]], device="cuda", dtype=torch.float32)

ee_pose_3 = torch.tensor([[ 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device='cuda:0', dtype=torch.float32)
ee_pose_2 = torch.tensor([[ 0.758936697969954, 0, 0.09210183252246132, -7.9714e-08,  1.0000e+00, -1.3333e-07, -4.3204e-07,  0.0000e+00]], device='cuda:0', dtype=torch.float32)
ee_pose = torch.tensor([[ 3.5970e-01,  5.0284e-08,  3.1842e-01,  -7.9714e-08, -1.3333e-07, -4.3204e-07, 1.0000e+00, 0.0000e+00]], device='cuda:0', dtype=torch.float32)


while True:
    action = sample_random_action(joint_limits, batch_size=1)

    input_joints = action

    obs, rew, term, trunc, info = env.step(ee_pose, expensive=True)
    arm_action = env.unwrapped.action_manager._terms["arm"]

    # OSC tracks these internally after each step
    print("EE pos (robot base frame):", arm_action._ee_pose_b)   # (N, 7)
    
    robot = env.scene["robot"]

    print(obs["policy"]["ee_pose"][0])
    output_joints = obs["policy"]["arm_joint_pos"][0]
    
    frames.append(obs["splat"]["cam0"])
    
    frames_2.append(obs["splat"]["cam1"])
    wrist_frames.append(obs["splat"]["wrist_cam"])

    step += 1

    if step >= max_steps:
        print(f"\nCollected {step} samples.")
        break


frames = np.stack(frames, axis=0)
iio.imwrite("scene_viz.mp4", frames, fps=30)

wrist_frames = np.stack(wrist_frames, axis=0)
iio.imwrite("scene_viz_wrist.mp4", wrist_frames, fps=30)

frames_2 = np.stack(frames_2, axis=0)
iio.imwrite("scene_viz_2.mp4", frames_2, fps=30)

env.close()
simulation_app.close()