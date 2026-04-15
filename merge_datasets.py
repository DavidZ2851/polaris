#!/usr/bin/env python3
"""
Merge multiple episode dataset folders into a single folder by moving/renaming episodes.

Usage:
    python merge_datasets.py --input /path/to/dataset1 /path/to/dataset2 --output /path/to/merged

Example:
    python merge_datasets.py --input pick_place_red_mug_30 pick_place_red_mug_50 --output pick_place_red_mug_80
"""

import os
import shutil
import argparse
from glob import glob
from tqdm import tqdm


def get_episodes(dataset_dir):
    """Get sorted list of episode directories."""
    episodes = glob(os.path.join(dataset_dir, "episode_*"))
    episodes = sorted(episodes, key=lambda x: int(os.path.basename(x).split('_')[1]))
    return episodes


def merge_datasets(input_dirs, output_dir, dry_run=False):
    """
    Merge multiple dataset folders into one by moving episodes.
    
    Args:
        input_dirs: List of input dataset directories
        output_dir: Output directory for merged dataset
        dry_run: If True, only print what would be done without moving
    """
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Merging {len(input_dirs)} datasets into: {output_dir}")
    print("-" * 60)
    
    episode_counter = 0
    
    for input_dir in input_dirs:
        episodes = get_episodes(input_dir)
        print(f"  {input_dir}: {len(episodes)} episodes")
    
    print("-" * 60)
    
    if dry_run:
        print("\n[DRY RUN] Would perform the following moves:\n")
    
    for input_dir in input_dirs:
        episodes = get_episodes(input_dir)
        
        for episode_path in tqdm(episodes, desc=f"Moving {os.path.basename(input_dir)}"):
            new_name = f"episode_{episode_counter:06d}"
            new_path = os.path.join(output_dir, new_name)
            
            if dry_run:
                print(f"  {episode_path} -> {new_path}")
            else:
                shutil.move(episode_path, new_path)
            
            episode_counter += 1
        
        # Delete empty input folder after moving all episodes
        if not dry_run:
            try:
                os.rmdir(input_dir)
                print(f"Deleted empty folder: {input_dir}")
            except OSError:
                print(f"Warning: Could not delete {input_dir} (not empty)")
        else:
            print(f"  [DRY RUN] Would delete empty folder: {input_dir}")
    
    print("-" * 60)
    print(f"{'[DRY RUN] Would move' if dry_run else 'Moved'} {episode_counter} episodes to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Merge episode dataset folders by moving")
    parser.add_argument("--input", "-i", nargs='+', required=True,
                        help="Input dataset directories to merge")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for merged dataset")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Show what would be done without actually moving")
    
    args = parser.parse_args()
    
    merge_datasets(args.input, args.output, args.dry_run)


if __name__ == "__main__":
    main()