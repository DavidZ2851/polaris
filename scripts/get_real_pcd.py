import cv2
import json
import numpy as np
from pathlib import Path
import av

DATA_DIR = Path("/home/haotian/.cache/huggingface/lerobot/Kovavavvavava/pick_place_red_mug_20260327_1")
CALIB_PATH = "/home/haotian/polaris/PolaRiS-Hub/put_red_cup_no_curtain/cam_calibration.json"

# Load calibration
with open(CALIB_PATH, 'r') as f:
    calib = json.load(f)

for cam in calib:
    calib[cam]["extrinsic"] = np.array(calib[cam]["extrinsic"])
    calib[cam]["intrinsic"] = np.array(calib[cam]["intrinsic"])

# Paths
front_color = DATA_DIR / "videos/chunk-000/observation.images.cam_azure_kinect_front.color/episode_000000.mp4"
front_depth = DATA_DIR / "videos/chunk-000/observation.images.cam_azure_kinect_front.transformed_depth/episode_000000.mkv"
left_color = DATA_DIR / "videos/chunk-000/observation.images.cam_azure_kinect_left.color/episode_000000.mp4"
left_depth = DATA_DIR / "videos/chunk-000/observation.images.cam_azure_kinect_left.transformed_depth/episode_000000.mkv"


def load_first_rgb(video_path):
    """Load first RGB frame using cv2."""
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def load_first_depth_mkv(video_path):
    """Load first 16-bit depth frame from MKV using PyAV."""
    container = av.open(str(video_path))
    video_stream = container.streams.video[0]
    
    for frame in container.decode(video_stream):
        # 16-bit grayscale depth
        frame_array = frame.to_ndarray(format='gray16le')
        container.close()
        # Convert mm -> meters
        return frame_array.astype(np.float32) / 1000.0
    
    container.close()
    return None


def depth_to_pointcloud_world(depth, rgb, intrinsic, cam_to_base):
    """
    Convert depth image to point cloud in world/base frame.
    """
    H, W = depth.shape
    
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    z = depth.flatten()
    
    # Valid depth mask
    valid = (z > 0.1) & (z < 3.0)
    u, v, z = u[valid], v[valid], z[valid]
    
    # Unproject: pixel -> camera frame
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    z_cam = z
    
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    
    # Camera -> Base (cam_to_base, direct transform)
    R = cam_to_base[:3, :3]
    t = cam_to_base[:3, 3]
    points_world = (R @ points_cam.T).T + t
    
    # Colors (BGR -> RGB)
    colors = rgb.reshape(-1, 3)[valid]
    colors = colors[:, ::-1]
    
    return points_world, colors


def save_ply(filename, points, colors):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(len(points)):
            f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} "
                    f"{int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}\n")


# Load data
rgb0 = load_first_rgb(front_color)
depth0 = load_first_depth_mkv(front_depth)
rgb1 = load_first_rgb(left_color)
depth1 = load_first_depth_mkv(left_depth)

print(f"RGB0: {rgb0.shape}, Depth0: {depth0.shape}")
print(f"RGB1: {rgb1.shape}, Depth1: {depth1.shape}")
print(f"Depth0 range: {depth0.min():.3f} ~ {depth0.max():.3f} m")
print(f"Depth1 range: {depth1.min():.3f} ~ {depth1.max():.3f} m")

# Build point clouds
pcd0, colors0 = depth_to_pointcloud_world(
    depth0, rgb0, 
    calib["cam0"]["intrinsic"], 
    calib["cam0"]["extrinsic"]
)
pcd1, colors1 = depth_to_pointcloud_world(
    depth1, rgb1, 
    calib["cam1"]["intrinsic"], 
    calib["cam1"]["extrinsic"]
)

print(f"\nCam0 points: {pcd0.shape}")
print(f"  x: {pcd0[:,0].min():.3f} ~ {pcd0[:,0].max():.3f}")
print(f"  y: {pcd0[:,1].min():.3f} ~ {pcd0[:,1].max():.3f}")
print(f"  z: {pcd0[:,2].min():.3f} ~ {pcd0[:,2].max():.3f}")

print(f"\nCam1 points: {pcd1.shape}")
print(f"  x: {pcd1[:,0].min():.3f} ~ {pcd1[:,0].max():.3f}")
print(f"  y: {pcd1[:,1].min():.3f} ~ {pcd1[:,1].max():.3f}")
print(f"  z: {pcd1[:,2].min():.3f} ~ {pcd1[:,2].max():.3f}")

# Merge and save
points = np.concatenate([pcd0, pcd1], axis=0)
colors = np.concatenate([colors0, colors1], axis=0)

save_ply("base_cam0.ply", pcd0, colors0)
save_ply("base_cam1.ply", pcd1, colors1)
save_ply("base_merged.ply", points, colors)

print(f"\nSaved: base_cam0.ply, base_cam1.ply, base_merged.ply")
print(f"Total points: {len(points)}")