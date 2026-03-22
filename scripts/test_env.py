import torch
import argparse
import gymnasium as gym
from isaaclab.app import AppLauncher
import imageio.v3 as iio
import numpy as np

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
import omni.replicator.core as rep

env_cfg = parse_env_cfg(
    "DROID-PutRedCup-no-curtain",
    device="cuda",
    num_envs=1,
    use_fabric=True,
)

env: ManagerBasedRLSplatEnv = gym.make("DROID-PutRedCup-no-curtain", cfg=env_cfg)

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
obs, info = env.reset(object_positions=initial_conditions[0], expensive=True)

robot = env.scene["robot"]

step = 0
max_steps = 50
frames = [obs["splat"]["external_cam"]]
wrist_frames = [obs["splat"]["wrist_cam"]]


import torch

action_1 = torch.tensor([[0.01968304, -0.4766486 ,  0.00991582, -2.38033551,  0.01029314, 1.95322416, -0.00605614, 0]], device="cuda", dtype=torch.float32)
action_0 = torch.tensor([[0.0000, -0.6283,  0.0000, -2.5133,  0.0000,  1.8850,  0.0000,  0.000]], device="cuda", dtype=torch.float32)
while True:
    action = sample_random_action(joint_limits, batch_size=1)

    input_joints = action

    obs, rew, term, trunc, info = env.step(action_0, expensive=True)
    # obs, info = env.reset(object_positions=initial_conditions[0], expensive=True)

    output_joints = obs["policy"]["arm_joint_pos"][0]
    
    frames.append(obs["splat"]["external_cam"])
    wrist_frames.append(obs["splat"]["wrist_cam"])

    step += 1

    if step >= max_steps:
        print(f"\nCollected {step} samples.")
        break


frames = np.stack(frames, axis=0)
iio.imwrite("scene_viz.mp4", frames, fps=30)

wrist_frames = np.stack(wrist_frames, axis=0)
iio.imwrite("scene_viz_wrist.mp4", wrist_frames, fps=30)

env.close()
simulation_app.close()