import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Union, List
import glob

def run(cmd, cwd=None, env=None):
    print("[RUN]", " ".join(map(str, cmd)))
    p = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {p.returncode}: {' '.join(map(str, cmd))}\n\n"
            f"--- STDOUT ---\n{p.stdout}\n\n"
            f"--- STDERR ---\n{p.stderr}\n"
        )
    return p

def check_bin(name: str):
    if shutil.which(name) is None:
        raise FileNotFoundError(f"'{name}' not found on PATH. Install it and make sure it's in PATH.")


from pathlib import Path
import os
import shutil

from pathlib import Path
import os
import shutil

def run_colmap(
    images_dir: Path,
    work_dir: Path,
    matcher: str = "sequential",
    use_gpu: bool = False,
    undistort: bool = True,
) -> Path:
    """
    Runs COLMAP SfM for images_dir/*.jpg (or *.png), and optionally creates an
    UNDISTORTED COLMAP dataset (PINHOLE/SIMPLE_PINHOLE) compatible with splatting repos.

    Returns:
        If undistort=True: Path to undistorted dataset root:
            work_dir/undistorted/  (contains images/ and sparse/0/)
        Else: Path to the first sparse model dir:
            work_dir/sparse/<model_id>/
    """
    check_bin("colmap")

    images_dir = Path(images_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    if not any(p.is_file() and p.suffix in image_exts for p in images_dir.iterdir()):
        raise FileNotFoundError(f"No images found in {images_dir} (expected .jpg/.png).")

    # Headless-friendly
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    db_path = work_dir / "database.db"
    sparse_dir = work_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # NOTE: On many headless servers, COLMAP GPU SIFT fails due to OpenGL context issues.
    sift_gpu = "1" if use_gpu else "0"

    def sparse_model_exists() -> bool:
        """Detect whether a COLMAP sparse model exists under work_dir/sparse/*."""
        if not sparse_dir.exists():
            return False
        for m in sorted([p for p in sparse_dir.iterdir() if p.is_dir()]):
            if (m / "cameras.bin").exists() or (m / "cameras.txt").exists():
                return True
        return False

    def first_model_dir() -> Path:
        model_dirs = sorted([p for p in sparse_dir.iterdir() if p.is_dir()])
        if not model_dirs:
            raise RuntimeError(f"No sparse model produced for {images_dir}")
        return model_dirs[0]

    # 1) Feature extraction (skip if database exists and non-empty)
    if db_path.exists() and db_path.stat().st_size > 0:
        print(f"[COLMAP] Skip feature_extractor: {db_path} exists.")
    else:
        run([
            "colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "1",
        ])

    # 2) Matching (best-effort skip if sparse already exists)
    if sparse_model_exists():
        print("[COLMAP] Skip matcher: sparse model already exists.")
    else:
        if matcher == "exhaustive":
            run([
                "colmap", "exhaustive_matcher",
                "--database_path", str(db_path),
            ])
        elif matcher == "sequential":
            run([
                "colmap", "sequential_matcher",
                "--database_path", str(db_path),
                # "--SequentialMatching.overlap", "10",
            ])
        else:
            raise ValueError("matcher must be 'exhaustive' or 'sequential'")

    # 3) Sparse reconstruction (mapper)
    if sparse_model_exists():
        print("[COLMAP] Skip mapper: sparse model already exists.")
    else:
        run([
            "colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
            "--log_to_stderr", "1",
        ])

    model_dir = first_model_dir()
    print(f"[COLMAP] Sparse model -> {model_dir}")

    if not undistort:
        return model_dir

    # 4) Undistort images + produce undistorted COLMAP model
    undist_root = work_dir / "undistorted"
    undist_root.mkdir(parents=True, exist_ok=True)

    # COLMAP will create:
    #   undist_root/images/   (undistorted images)
    #   undist_root/sparse/   (may contain cameras/images/points3D, sometimes directly)
    #   undist_root/stereo/   (optional)
    print(f"[COLMAP] Undistorting -> {undist_root}")
    run([
        "colmap", "image_undistorter",
        "--image_path", str(images_dir),
        "--input_path", str(model_dir),
        "--output_path", str(undist_root),
        "--output_type", "COLMAP",
    ])

    # Normalize sparse output to undist_root/sparse/0 for downstream tools
    sparse_out = undist_root / "sparse"
    sparse0 = sparse_out / "0"
    sparse0.mkdir(parents=True, exist_ok=True)

    # Some COLMAP versions output cameras/images/points3D directly under undist_root/sparse/
    # Move them into sparse/0 if needed.
    for fname in ["cameras.bin", "images.bin", "points3D.bin", "cameras.txt", "images.txt", "points3D.txt"]:
        src = sparse_out / fname
        if src.exists():
            dst = sparse0 / fname
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))

    # If still not present, maybe COLMAP already made sparse/0 (or multiple models).
    # Ensure at least one model directory exists.
    if not any((sparse0 / f).exists() for f in ["cameras.bin", "cameras.txt"]):
        # Try to detect a model dir under undist_root/sparse/*
        model_dirs = sorted([p for p in sparse_out.iterdir() if p.is_dir()])
        if model_dirs:
            # If it's not "0", you can either return that or copy it to 0.
            # We'll copy to sparse/0 for consistency.
            src_model = model_dirs[0]
            if src_model != sparse0:
                for item in src_model.iterdir():
                    dst = sparse0 / item.name
                    if dst.exists():
                        if dst.is_dir():
                            shutil.rmtree(dst)
                        else:
                            dst.unlink()
                    if item.is_dir():
                        shutil.copytree(item, dst)
                    else:
                        shutil.copy2(item, dst)

    # Final sanity check
    if not (sparse0 / "cameras.bin").exists() and not (sparse0 / "cameras.txt").exists():
        raise RuntimeError(
            f"[COLMAP] Undistortion completed, but no cameras model found in {sparse0}. "
            f"Check COLMAP output under {undist_root}."
        )

    print(f"[COLMAP] Undistorted dataset ready -> {undist_root}")
    return undist_root

def main():
    parser = argparse.ArgumentParser(description="Run COLMAP dense reconstruction from a video.")
    
    parser.add_argument(
        "--video",
        help="Path to input video (mp4/mov/etc).",
        type=str,
        required=True,
    )
    parser.add_argument("--out", default="/home/haotian/polaris/video_outputs", help="Output working directory.")
    parser.add_argument("--fps", type=float, default=10.0, help="Frame extraction FPS (e.g., 5-15).")
    args = parser.parse_args()

    video_name = Path(args.video).stem
    out_dir = Path(f"{args.out}/{video_name}") 

    out_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Check if frames already exist
    existing_frames = list(frames_dir.glob("frame_*.jpg"))

    if len(existing_frames) > 0:
        print(f"Frames already exist in {frames_dir}. Skipping ffmpeg extraction.")
    else:
        run([
            "ffmpeg",
            "-i", str(args.video),
            "-vf", f"fps={args.fps}",
            str(frames_dir / "frame_%04d.jpg")
        ])
        print(f"Extracted frames saved in {frames_dir}")

    # Run COLMAP on the extracted frames
    fused_ply = run_colmap(frames_dir, out_dir, use_gpu=True)

if __name__ == "__main__":
    main()



