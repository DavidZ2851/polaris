"""
Compute success/progress statistics from eval_results.csv.

Usage:
    python polaris/scripts/eval_summary.py <run_folder>
    python polaris/scripts/eval_summary.py runs/diffusion_policy_human=5_robot=5
"""

import argparse
import pandas as pd
from pathlib import Path


def summarize(csv_path: Path):
    df = pd.read_csv(csv_path)
    n = len(df)
    if n == 0:
        print("No episodes found.")
        return

    s1 = (df["progress"] >= 1/3 - 1e-6).sum()
    s2 = (df["progress"] >= 2/3 - 1e-6).sum()
    s3 = (df["success"] == True).sum()
    avg_progress = df["progress"].mean()

    print(f"Run:      {csv_path.parent}")
    print(f"Episodes: {n}")
    print(f"Avg progress (mean across episodes): {avg_progress:.3f}  ({100*avg_progress:.1f}%)")
    print(f"Stage 1 (reach,  progress≥0.33): {s1:3d}/{n}  ({100*s1/n:.1f}%)")
    print(f"Stage 2 (lift,   progress≥0.67): {s2:3d}/{n}  ({100*s2/n:.1f}%)")
    print(f"Stage 3 (place,  success=True  ): {s3:3d}/{n}  ({100*s3/n:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_folder", type=str, help="Path to run folder containing eval_results.csv")
    args = parser.parse_args()

    csv_path = Path(args.run_folder) / "eval_results.csv"
    if not csv_path.exists():
        print(f"Not found: {csv_path}")
        return

    summarize(csv_path)


if __name__ == "__main__":
    main()
