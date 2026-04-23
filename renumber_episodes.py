#!/usr/bin/env python3
"""
renumber_episodes.py
Renumbers subfolders named episode_NNNNNN under a given path
so they are sequential with no gaps (episode_000000, episode_000001, ...).

Usage:
    python renumber_episodes.py /path/to/your/folder
    python renumber_episodes.py /path/to/your/folder --dry-run
"""

import os
import re
import sys
import argparse


EPISODE_PATTERN = re.compile(r'^episode_(\d+)$', re.IGNORECASE)


def get_episode_folders(root: str) -> list[tuple[int, str]]:
    """Return sorted list of (number, full_path) for matching episode folders."""
    entries = []
    try:
        for name in os.listdir(root):
            m = EPISODE_PATTERN.match(name)
            if m and os.path.isdir(os.path.join(root, name)):
                entries.append((int(m.group(1)), name))
    except FileNotFoundError:
        print(f"ERROR: Path not found: {root}")
        sys.exit(1)
    except PermissionError:
        print(f"ERROR: Permission denied: {root}")
        sys.exit(1)

    entries.sort(key=lambda x: x[0])
    return entries


def renumber(root: str, dry_run: bool = False) -> None:
    episodes = get_episode_folders(root)

    if not episodes:
        print("No episode_NNNNNN folders found.")
        return

    print(f"Found {len(episodes)} episode folder(s) in: {root}")
    if dry_run:
        print("DRY RUN — no files will be renamed.\n")

    # Check if already sequential
    numbers = [num for num, _ in episodes]
    expected = list(range(0, len(episodes)))  # 0-indexed: 0, 1, 2, ...
    if numbers == expected:
        print("Folders are already numbered sequentially. Nothing to do.")
        return

    # We rename in two passes to avoid collisions
    # Pass 1: rename everything to a safe temporary name
    temp_names = []
    for i, (num, name) in enumerate(episodes):
        src = os.path.join(root, name)
        tmp = os.path.join(root, f"__tmp_episode_{i:06d}__")
        print(f"  [pass 1] {name}  →  {os.path.basename(tmp)}")
        if not dry_run:
            os.rename(src, tmp)
        temp_names.append(tmp)

    # Pass 2: rename temporaries to final sequential names
    print()
    for new_num, tmp in enumerate(temp_names, start=0):  # start=0
        final_name = f"episode_{new_num:06d}"
        dst = os.path.join(root, final_name)
        old_num = episodes[new_num][0]
        label = f"episode_{old_num:06d}"
        print(f"  [pass 2] {label}  →  {final_name}")
        if not dry_run:
            os.rename(tmp, dst)

    print()
    if dry_run:
        print("Dry run complete. Run without --dry-run to apply changes.")
    else:
        print(f"Done. {len(episodes)} folder(s) renumbered sequentially.")


def main():
    parser = argparse.ArgumentParser(
        description="Renumber episode_NNNNNN subfolders to be sequential (0-indexed)."
    )
    parser.add_argument("path", help="Path to the parent folder containing episode folders")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without renaming anything",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.path)
    if not os.path.isdir(root):
        print(f"ERROR: Not a directory: {root}")
        sys.exit(1)

    renumber(root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()