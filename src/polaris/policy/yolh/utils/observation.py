"""Build inference point clouds from RGB-D observations."""

import numpy as np

from .constants import IMG_MEAN, IMG_STD
from .transformation import rgbd_to_points


def build_observation_cloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Build the point cloud used for policy inference."""
    voxel_size = cfg["voxel_size"]
    ws_min = cfg["workspace_min"]
    ws_max = cfg["workspace_max"]

    coords, colors = rgbd_to_points(rgb, depth, intrinsic)
    if len(coords) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    in_ws = np.all((coords >= ws_min) & (coords <= ws_max), axis=1)
    coords = coords[in_ws]
    colors = colors[in_ws]
    if len(coords) == 0:
        return np.zeros((0, 6), dtype=np.float32)

    vi = np.floor(coords / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(vi, axis=0, return_index=True)
    coords = coords[unique_idx]
    colors = colors[unique_idx]

    colors = (colors - IMG_MEAN) / IMG_STD

    cloud = np.concatenate([coords, colors], axis=-1).astype(np.float32)
    return cloud
