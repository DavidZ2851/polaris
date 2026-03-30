"""
Process a PolaRiS / Isaac Lab dataset into per-timestep npz files.

State-action pairing (set by collect_data.py):
    ┌───────┬──────────┬─────────────┐
    │ index │ state    │ abs_action  │
    ├───────┼──────────┼─────────────┤
    │   0   │ s0       │ a0          │  ← (s0, a0) paired correctly
    │   1   │ s1       │ a1          │
    │  ...  │ ...      │ ...         │
    │  T-1  │ s_{T-1}  │ a_{T-1}     │
    │   T   │ s_T      │ None        │  ← terminal, no action
    └───────┴──────────┴─────────────┘


Input layout:
    <dataset_root>/
        episode_000000/
            trajectory.npz
                states       (T+1, 8)   [pos(3) | wxyz(4) | gripper(1)]
                abs_actions  (T,   8)   [pos(3) | wxyz(4) | gripper(1)]
                gripper_pcds (T+1, 4, 3)
                goal_gripper_pcds    (T+1, 4, 3)
                object_pcd   (T+1, N, 3) or (N, 3)
        episode_000001/
            ...

Output layout:
    <output_root>/
        demo_0/
            0.npz
            1.npz
            ...
        demo_1/
            0.npz
            ...

Per-timestep npz keys:
    state            (1, 10)      eef pos(3) + rot6(4) + gripper_width(1)
    point_cloud      (1, N, 3)    object point cloud
    action           (1, 10)      delta_pos(3) + delta_rot6d(6) + delta_gripper(1)
    gripper_pcd      (1, 4, 3)    current gripper 4-point pcd
    goal_gripper_pcd (1, 4, 3)    next subgoal gripper pcd

Delta action definition:
    delta_pos     = abs_action[:3]  - state[:3]          (world/base frame)
    delta_rot6d   = rot6d(R_state.T @ R_action)          (local EEF frame)
    delta_gripper = abs_action[9]   - state[9]


Usage:
    python process_polaris_to_npz.py --dataset /path/to/root --output /path/to/out
    python process_polaris_to_npz.py --dataset /path/to/root --output /path/to/out --n 10
"""

import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm


import robosuite
import robosuite.utils.transform_utils as T
from polaris.utils_.transform_utils import rotation_transfer_matrix_to_6D, quat_wxyz_to_rot6d, state_quat_to_rot6d, compute_delta_action
# ---------------------------------------------------------------------------
# Episode processing
# ---------------------------------------------------------------------------


def filter_stationary(states: np.ndarray, abs_actions: np.ndarray,
                       gripper_pcd: np.ndarray, goal_gripper_pcd: np.ndarray,
                       object_pcd: np.ndarray,
                       pos_thresh: float = 0.002,
                       max_stationary_steps: int = 5) -> tuple:
    """
    Remove timesteps where the robot is stationary for too long.
    Stationary is defined as EE not moving between consecutive states.

    Parameters
    ----------
    states        : (T+1, 8)
    abs_actions   : (T,   8)
    pos_thresh    : movement below this (meters) counts as stationary
    max_stationary_steps : remove runs of stationary steps longer than this
    """
    T = abs_actions.shape[0]

    # Displacement from state[t] to state[t+1]
    pos_delta = np.linalg.norm(states[1:T+1, :3] - states[:T, :3], axis=1)  # (T,)

    is_stationary = pos_delta < pos_thresh

    keep = np.ones(T, dtype=bool)
    run_start = None
    for t in range(T):
        if is_stationary[t]:
            if run_start is None:
                run_start = t
        else:
            if run_start is not None:
                if (t - run_start) > max_stationary_steps:
                    keep[run_start:t] = False
                run_start = None

    if run_start is not None:
        if (T - run_start) > max_stationary_steps:
            keep[run_start:] = False

    kept_idx = np.where(keep)[0]
    print(f"  filter_stationary: keeping {keep.sum()}/{T} steps "
          f"(removed {(~keep).sum()} stationary steps)")

    state_idx = np.concatenate([kept_idx, [kept_idx[-1] + 1]])
    state_idx = np.clip(state_idx, 0, len(states) - 1)

    return (
        states[state_idx],
        abs_actions[kept_idx],
        gripper_pcd[state_idx],
        goal_gripper_pcd[state_idx],
        object_pcd[state_idx] if object_pcd.ndim == 3 else object_pcd,
    )


def process_episode(ep_dir: str, out_ep_dir: str) -> int:
    """
    Process one episode folder into per-timestep npz files.
    Returns number of timesteps written, or 0 on failure.
    """
    traj_path = os.path.join(ep_dir, "trajectory.npz")
    if not os.path.exists(traj_path):
        print(f"  [warn] trajectory.npz not found in {ep_dir} — skipping.")
        return 0

    traj = dict(np.load(traj_path, allow_pickle=False))

    # ── Validate ──────────────────────────────────────────────────────────

    required = {"states", "abs_actions", "gripper_pcds", "goal_pcds", "object_pcd"}
    missing  = required - set(traj.keys())
    if missing:
        print(f"  [warn] {ep_dir}: missing keys {missing} — skipping.")
        return 0

    states      = traj["states"].astype(np.float32)        # (T+1, 10)
    abs_actions = traj["abs_actions"].astype(np.float32)   # (T,   10)
    gripper_pcd = traj["gripper_pcds"].astype(np.float32) # (T+1, 4, 3)
    goal_gripper_pcd   = traj["goal_pcds"].astype(np.float32)     # (T+1, 4, 3)
    object_pcd  = traj["object_pcd"].astype(np.float32)    # (T+1, N, 3) or (N, 3)

    # Filter action
    states, abs_actions, gripper_pcd, goal_gripper_pcd, object_pcd = filter_stationary(
            states, abs_actions, gripper_pcd, goal_gripper_pcd, object_pcd,
            pos_thresh=0.002,
            max_stationary_steps=5,
    )
    T = abs_actions.shape[0]

    if states.shape[0] != T + 1:
        print(f"  [warn] {ep_dir}: states={states.shape[0]} expected {T+1} — skipping.")
        return 0

    # ── object_pcd: handle per-step (T+1,N,3) or static (N,3) ────────────
    if object_pcd.ndim == 2:
        # static — tile to (T+1, N, 3)
        object_pcd = np.tile(object_pcd[None], (T + 1, 1, 1))

    os.makedirs(out_ep_dir, exist_ok=True)

    # ── Write per-timestep npz ────────────────────────────────────────────
    # obs[t] = states[t], action[t] = abs_actions[t]  for t = 0..T-1
    # states[T] = terminal state, not written as a transition
    for t in range(T):
        delta_action = compute_delta_action(states[t], abs_actions[t], clip_action=True) 
        
        np.savez_compressed(
            os.path.join(out_ep_dir, f"{t}.npz"),
            state            = state_quat_to_rot6d(states[t])[None, :],   # (8,) -> (1, 10)
            point_cloud      = object_pcd[t][None, :],    # (1, N, 3)
            action           = delta_action[None, :],     # (1, 10)
            gripper_pcd      = gripper_pcd[t][None, :],  # (1, 4, 3)
            goal_gripper_pcd = goal_gripper_pcd[t][None, :],     # (1, 4, 3)
        )

    return T


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    dataset_root = args.dataset
    output_root  = args.output
    n_limit      = args.n

    ep_dirs = sorted(glob(os.path.join(dataset_root, "episode_*")))
    ep_dirs = [d for d in ep_dirs if os.path.isdir(d)]

    if not ep_dirs:
        raise RuntimeError(f"No episode_* directories found under {dataset_root}")
    if n_limit is not None:
        ep_dirs = ep_dirs[:n_limit]

    print(f"Found {len(ep_dirs)} episodes under {dataset_root}")
    print(f"Output root : {output_root}\n")

    os.makedirs(output_root, exist_ok=True)

    total_steps   = 0
    total_demos   = 0
    demo_idx      = 0  # only incremented on success

    for ep_dir in tqdm(ep_dirs, desc="Processing"):
        out_ep_dir = os.path.join(output_root, f"demo_{demo_idx}")

        
        n_steps = process_episode(ep_dir, out_ep_dir)

        if n_steps == 0:
            continue  # skipped, don't increment demo_idx

        total_steps += n_steps
        total_demos += 1
        tqdm.write(f"  demo_{demo_idx:04d}  ({ep_dir})  →  {n_steps} steps")
        demo_idx += 1

    print(f"\n{'='*50}")
    print(f"Demos written : {total_demos}")
    print(f"Total steps   : {total_steps}")
    print(f"Output        : {output_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PolaRiS episodes into per-timestep npz files."
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Root directory containing episode_XXXXXX/ subdirectories.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output root directory for demo_N/t.npz files.",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Only process the first N episodes (useful for debugging).",
    )
    args = parser.parse_args()
    main(args)