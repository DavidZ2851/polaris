import tyro
import mediapy

# import wandb
import tqdm
import gymnasium as gym
import torch
import argparse
import pandas as pd


from pathlib import Path
from isaaclab.app import AppLauncher

from polaris.config import EvalArgs
import numpy as np

TABLE_RANGE = {
    "x": (0.383, 0.45891),
    "y": (0.0, 0.12889),
    # "z" : (0.15, 0.15),
    "z": (0.11022, 0.11022),
}

def random_cup_pose():
    return [
        np.random.uniform(*TABLE_RANGE["x"]),
        np.random.uniform(*TABLE_RANGE["y"]),
        np.random.uniform(*TABLE_RANGE["z"]),
    ]


def main(eval_args: EvalArgs):
    # This must be done before importing anything from IsaacLab
    # Inside main function to avoid launching IsaacLab in global scope
    # >>>> Isaac Sim App Launcher <<<<
    parser = argparse.ArgumentParser()
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = eval_args.headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    # >>>> Isaac Sim App Launcher <<<<

    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
    from polaris.environments.manager_based_rl_splat_environment import (
        ManagerBasedRLSplatEnv,
    )
    from polaris.utils import load_eval_initial_conditions
    from polaris.policy import InferenceClient
    # from real2simeval.autoscoring import TASK_TO_SUCCESS_CHECKER

    env_cfg = parse_env_cfg(
        eval_args.environment,
        device="cuda",
        num_envs=1,
        use_fabric=True,
    )
    env: MangerBasedRLSplatEnv = gym.make(eval_args.environment, cfg=env_cfg)  # type: ignore

    language_instruction, initial_conditions = load_eval_initial_conditions(
        usd=env.usd_file,
        initial_conditions_file=eval_args.initial_conditions_file,
        rollouts=eval_args.rollouts,
    )
    # Randomise cup pose

    cup_pos = random_cup_pose()
    ic = dict(initial_conditions[0])
    ic["red_cup"][:3] = cup_pos

    rollouts = eval_args.rollouts
    # Resume CSV logging
    run_folder = Path(eval_args.run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)
    csv_path = run_folder / "eval_results.csv"
    if csv_path.exists():
        episode_df = pd.read_csv(csv_path)
    else:
        episode_df = pd.DataFrame(
            {
                "episode": pd.Series(dtype="int"),
                "episode_length": pd.Series(dtype="int"),
                "success": pd.Series(dtype="bool"),
                "progress": pd.Series(dtype="float"),
            }
        )
    episode = len(episode_df)
    if episode >= rollouts:
        print("All rollouts have been evaluated. Exiting.")
        env.close()
        simulation_app.close()
        return

    policy_client: InferenceClient = InferenceClient.get_client(eval_args.policy)

    video = []
    horizon = env.max_episode_length
    bar = tqdm.tqdm(range(horizon))
    obs, info = env.reset(
        object_positions=ic, expensive=True
    )
    policy_client.reset()

    print(f" >>> Starting eval job from episode {episode + 1} of {rollouts} <<< ")
    while True:
        action, viz = policy_client.infer(obs, language_instruction, return_viz=True)
        if viz is not None:
            video.append(viz)
        obs, rew, term, trunc, info = env.step(
            torch.tensor(action).reshape(1, -1), expensive=True
        )

        bar.update(1)
        if term[0] or trunc[0] or bar.n >= horizon:
            policy_client.reset()

            # Save video and metadata
            filename = run_folder / f"episode_{episode}.mp4"
            mediapy.write_video(filename, video, fps=15)

            # Log episode results to CSV
            episode_data = {
                "episode": episode,
                "episode_length": bar.n,
                "success": info["rubric"]["success"],
                "progress": info["rubric"]["progress"],
            }
            episode_df = pd.concat(
                [episode_df, pd.DataFrame([episode_data])], ignore_index=True
            )
            episode_df.to_csv(csv_path, index=False)

            bar.close()
            print(f"Episode {episode} finished. Episode length: {bar.n}")
            episode += 1
            bar = tqdm.tqdm(range(horizon))
            cup_pos = random_cup_pose()
            ic = dict(initial_conditions[0])
            ic["red_cup"][:3] = cup_pos
            obs, info = env.reset(
                object_positions=ic, expensive = True
            )

            video = []
            if episode >= rollouts:
                break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)
