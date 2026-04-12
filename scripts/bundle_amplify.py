"""
Bundle separate AMPLIFY checkpoints (motion_tokenizer + forward_dynamics + inverse_dynamics)
into a single AMPLIFY checkpoint for faster inference server startup.

Usage:
    conda activate amplify
    cd /home/veraxiao/vxiao/human2robot/benchmark_new/AMPLIFY
    python ../polaris/scripts/bundle_amplify.py \\
        --mt_ckpt  ../polaris/policy_ckpt/amplify/motion_tokenizer/motion_human0_robot40/latest.pt \\
        --fd_ckpt  ../polaris/policy_ckpt/amplify/forward_dynamics/pick_mug_human0_robot40/latest.pt \\
        --id_ckpt  ../polaris/policy_ckpt/amplify/inverse_dynamics/id_pick_mug_human0_robot40_seed_0/latest.pt \\
        --save_to  ../polaris/policy_ckpt/amplify/bundled/pick_mug_human0_robot40.pt

Or with a different AMPLIFY repo (e.g. full-resolution variant):
    python ../polaris/scripts/bundle_amplify.py \\
        --amplify_root /path/to/other/AMPLIFY \\
        --mt_ckpt  ../polaris/policy_ckpt/amplify_noimageresize/robot_mt_fd100/motion.pt \\
        --fd_ckpt  ../polaris/policy_ckpt/amplify_noimageresize/robot_mt_fd100/forward.pt \\
        --id_ckpt  ../polaris/policy_ckpt/amplify_noimageresize/robot_mt_fd100/inverse.pt \\
        --save_to  ../polaris/policy_ckpt/amplify_noimageresize/robot_mt_fd100/amplify.pt
"""

import argparse
import os
import sys
import pathlib

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--amplify_root", type=str, default=None)
_pre_args, _ = _pre.parse_known_args()

ROOT_DIR = _pre_args.amplify_root or str(pathlib.Path(__file__).parent.parent.parent / "AMPLIFY")
sys.path.insert(0, ROOT_DIR)

from amplify import AMPLIFY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amplify_root", type=str, default=None, help="Path to AMPLIFY repo root (default: ../../../AMPLIFY relative to this script)")
    parser.add_argument("--mt_ckpt",  type=str, required=True, help="Motion tokenizer checkpoint")
    parser.add_argument("--fd_ckpt",  type=str, required=True, help="Forward dynamics checkpoint")
    parser.add_argument("--id_ckpt",  type=str, required=True, help="Inverse dynamics checkpoint")
    parser.add_argument("--save_to",  type=str, required=True, help="Output path for bundled checkpoint")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.save_to)), exist_ok=True)

    print(f"Using AMPLIFY repo: {ROOT_DIR}")
    print(f"Bundling AMPLIFY checkpoints:")
    print(f"  MT: {args.mt_ckpt}")
    print(f"  FD: {args.fd_ckpt}")
    print(f"  ID: {args.id_ckpt}")

    _, save_path = AMPLIFY.bundle(
        motion_tokenizer_ckpt=args.mt_ckpt,
        forward_dynamics_ckpt=args.fd_ckpt,
        inverse_dynamics_ckpt=args.id_ckpt,
        save_to=args.save_to,
    )
    print(f"Saved bundled checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
