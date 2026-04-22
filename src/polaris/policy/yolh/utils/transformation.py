"""Transformation and action utilities."""

import functools
import numpy as np


def _get_torch():
    import torch
    return torch


def _get_ptc():
    import pytorch3d.transforms.rotation_conversions as ptc
    return ptc


def _get_rotation_utils():
    from . import rotation_utils as rtu
    return rtu


def rpy_to_rot(rpy: np.ndarray, dtype=np.float64) -> np.ndarray:
    rpy = np.asarray(rpy, dtype=dtype)
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=dtype)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=dtype)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=dtype)
    return rz @ ry @ rx


def make_transform(xyz: np.ndarray, rpy: np.ndarray, dtype=np.float64) -> np.ndarray:
    transform = np.eye(4, dtype=dtype)
    transform[:3, :3] = rpy_to_rot(rpy, dtype=dtype)
    transform[:3, 3] = np.asarray(xyz, dtype=dtype)
    return transform


def parse_origin(element, dtype=np.float64):
    origin = element.find("origin") if element is not None else None
    if origin is None:
        zeros = np.zeros(3, dtype=dtype)
        return zeros, zeros.copy()

    xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ", dtype=dtype)
    rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ", dtype=dtype)
    if xyz.size != 3:
        xyz = np.zeros(3, dtype=dtype)
    if rpy.size != 3:
        rpy = np.zeros(3, dtype=dtype)
    return xyz, rpy


def rot_z_transform(angle: float, dtype=np.float64) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    transform = np.eye(4, dtype=dtype)
    transform[0, 0] = c
    transform[0, 1] = -s
    transform[1, 0] = s
    transform[1, 1] = c
    return transform


def rgbd_to_points(rgb, depth, intrinsic, mask=None):
    """Back-project an RGB-D frame into a point cloud."""
    fx, fy, cx, cy = intrinsic
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    depth_m = depth.astype(np.float32) / 1000.0
    valid = depth_m > 0

    # Exclude masked pixels.
    if mask is not None:
        valid = valid & (~mask)

    z = depth_m[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy

    coords = np.stack([x, y, z], axis=-1).astype(np.float32)
    colors = rgb[valid].astype(np.float32) / 255.0
    return coords, colors


def rgbd_to_points_masked(rgb, depth, intrinsic, mask):
    """Compatibility wrapper for masked RGB-D back-projection."""
    return rgbd_to_points(rgb, depth, intrinsic, mask=mask)

VALID_ROTATION_REPRESENTATIONS = [
    'axis_angle',
    'euler_angles',
    'quaternion',
    'matrix',
    'rotation_6d',
    'rotation_9d',
    'rotation_10d'
]
ROTATION_REPRESENTATION_DIMS = {
    'axis_angle': 3,
    'euler_angles': 3,
    'quaternion': 4,
    'matrix': 9,
    'rotation_6d': 6,
    'rotation_9d': 9,
    'rotation_10d': 10
}


def rotation_transform(
    rot,
    from_rep, 
    to_rep, 
    from_convention = None, 
    to_convention = None
):
    """Convert one rotation representation to another."""
    assert from_rep in VALID_ROTATION_REPRESENTATIONS, "Invalid rotation representation: {}".format(from_rep)
    assert to_rep in VALID_ROTATION_REPRESENTATIONS, "Invalid rotation representation: {}".format(to_rep)
    if from_rep == 'euler_angles':
        assert from_convention is not None
    else:
        from_convention = None
    if to_rep == 'euler_angles':
        assert to_convention is not None
    else:
        to_convention = None

    if from_rep == to_rep and from_convention == to_convention:
        return rot

    torch = _get_torch()

    if from_rep != "matrix":
        if from_rep in ['rotation_9d', 'rotation_10d']:
            rtu = _get_rotation_utils()
            to_mat = getattr(rtu, "{}_to_matrix".format(from_rep))
        else:
            ptc = _get_ptc()
            to_mat = getattr(ptc, "{}_to_matrix".format(from_rep))
            if from_convention is not None:
                to_mat = functools.partial(to_mat, convention = from_convention)
        mat = to_mat(torch.from_numpy(rot)).numpy()
    else:
        mat = rot
        
    if to_rep != "matrix":
        if to_rep in ['rotation_9d', 'rotation_10d']:
            rtu = _get_rotation_utils()
            to_ret = getattr(rtu, "matrix_to_{}".format(to_rep))
        else:
            ptc = _get_ptc()
            to_ret = getattr(ptc, "matrix_to_{}".format(to_rep))
            if to_convention is not None:
                to_ret = functools.partial(to_ret, convention = to_convention)
        ret = to_ret(torch.from_numpy(mat)).numpy()
    else:
        ret = mat
    
    return ret


def xyz_rot_transform(
    xyz_rot,
    from_rep, 
    to_rep, 
    from_convention = None, 
    to_convention = None
):
    """
    Transform an xyz_rot representation into another equivalent xyz_rot representation.
    """
    assert from_rep in VALID_ROTATION_REPRESENTATIONS, "Invalid rotation representation: {}".format(from_rep)
    assert to_rep in VALID_ROTATION_REPRESENTATIONS, "Invalid rotation representation: {}".format(to_rep)

    if from_rep == to_rep and from_convention == to_convention:
        return xyz_rot

    xyz_rot = np.array(xyz_rot)
    if from_rep != "matrix":
        assert xyz_rot.shape[-1] == 3 + ROTATION_REPRESENTATION_DIMS[from_rep]
        xyz = xyz_rot[..., :3]
        rot = xyz_rot[..., 3:]
    else:
        assert xyz_rot.shape[-1] == 4 and xyz_rot.shape[-2] == 4
        xyz = xyz_rot[..., :3, 3]
        rot = xyz_rot[..., :3, :3]
    rot = rotation_transform(
        rot = rot,
        from_rep = from_rep,
        to_rep = to_rep,
        from_convention = from_convention,
        to_convention = to_convention
    )
    if to_rep != "matrix":
        return np.concatenate((xyz, rot), axis = -1)
    else:
        res = np.zeros(xyz.shape[:-1] + (4, 4), dtype = np.float32)
        res[..., :3, :3] = rot
        res[..., :3, 3] = xyz
        res[..., 3, 3] = 1
        return res


def xyz_rot_to_mat(xyz_rot, rotation_rep, rotation_rep_convention = None):
    """
    Transform an xyz_rot representation under any rotation form to an unified 4x4 pose representation.
    """
    return xyz_rot_transform(
        xyz_rot,
        from_rep = rotation_rep,
        to_rep = "matrix",
        from_convention = rotation_rep_convention
    )


def mat_to_xyz_rot(mat, rotation_rep, rotation_rep_convention = None):
    """
    Transform an unified 4x4 pose representation to an xyz_rot representation under any rotation form.
    """
    return xyz_rot_transform(
        mat,
        from_rep = "matrix",
        to_rep = rotation_rep,
        to_convention = rotation_rep_convention
    )


def apply_mat_to_pose(pose, mat, rotation_rep, rotation_rep_convention = None):
    """
    Apply transformation matrix mat to pose under any rotation form.
    """
    assert rotation_rep in VALID_ROTATION_REPRESENTATIONS, "Invalid rotation representation: {}".format(rotation_rep)
    mat = np.array(mat)
    pose = np.array(pose)
    assert mat.shape == (4, 4)
    if rotation_rep == "matrix":
        assert pose.shape[-2] == 4 and pose.shape[-1] == 4
        res_pose = mat @ pose
        return res_pose
    assert pose.shape[-1] == 3 + ROTATION_REPRESENTATION_DIMS[rotation_rep]
    pose_mat = xyz_rot_to_mat(
        xyz_rot = pose,
        rotation_rep = rotation_rep,
        rotation_rep_convention = rotation_rep_convention
    )
    res_pose_mat = mat @ pose_mat
    res_pose = mat_to_xyz_rot(
        mat = res_pose_mat,
        rotation_rep = rotation_rep,
        rotation_rep_convention = rotation_rep_convention
    )
    return res_pose


def apply_mat_to_pcd(pcd, mat):
    """
    Apply transformation matrix mat to point cloud.
    """
    mat = np.array(mat)
    assert mat.shape == (4, 4)
    pcd[..., :3] = (mat[:3, :3] @ pcd[..., :3].T).T + mat[:3, 3]
    return pcd


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert action rotation-6d representation to a 3x3 rotation matrix.

    This follows the project's row-encoding convention used in action data.
    """
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """Extract the project's row-encoded rotation-6d representation."""
    mat = np.asarray(mat, dtype=np.float64)
    return mat[..., :2, :].reshape(mat.shape[:-2] + (6,))


def unnormalize_action(action: np.ndarray, cfg: dict) -> np.ndarray:
    """Map action values from [-1, 1] back to physical units."""
    action = np.asarray(action, dtype=np.float64).copy()
    trans_min = np.asarray(cfg["trans_min"], dtype=np.float64)
    trans_max = np.asarray(cfg["trans_max"], dtype=np.float64)
    max_w = float(cfg["max_gripper_width"])
    if max_w <= 0:
        raise ValueError(f"max_gripper_width must be > 0, got {max_w}")

    action[..., :3] = (action[..., :3] + 1.0) / 2.0 * (trans_max - trans_min) + trans_min
    action[..., -1] = (action[..., -1] + 1.0) / 2.0 * max_w
    return action.astype(np.float32)


def project_action_to_base(action_cam: np.ndarray, cam_to_base: np.ndarray) -> np.ndarray:
    """Project an action chunk from camera frame to robot base frame."""
    action_cam = np.asarray(action_cam, dtype=np.float64)
    transform = np.asarray(cam_to_base, dtype=np.float64)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    out = action_cam.copy()
    out[..., :3] = action_cam[..., :3] @ rotation.T + translation

    rot_cam = np.swapaxes(rot6d_to_matrix(action_cam[..., 3:9]), -1, -2)
    rot_base = rotation @ rot_cam
    out[..., 3:9] = matrix_to_rot6d(rot_base)
    return out.astype(np.float32)


def rot6d_angular_distance(r1: np.ndarray, r2: np.ndarray) -> float:
    """Angular distance in radians between two rotation-6d vectors."""
    m1 = rot6d_to_matrix(r1)
    m2 = rot6d_to_matrix(r2)
    diff = m1 @ np.swapaxes(m2, -1, -2)
    cos_val = np.clip((np.trace(diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos_val))


def discretize_rotation(
    rot_begin: np.ndarray, rot_end: np.ndarray, step_size: float
) -> list:
    """Interpolate a rotation-6d command with bounded angular step size."""
    angle = rot6d_angular_distance(rot_begin, rot_end)
    n_steps = max(1, int(angle / step_size) + 1)
    steps = []
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        steps.append(rot_begin * (1.0 - alpha) + rot_end * alpha)
    return steps


def rot_mat_x_axis(angle):
    """
    3x3 transformation matrix for rotation along x axis.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype = np.float32)

def rot_mat_y_axis(angle):
    """
    3x3 transformation matrix for rotation along y axis.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype = np.float32)

def rot_mat_z_axis(angle):
    """
    3x3 transformation matrix for rotation along z axis.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype = np.float32)

def rot_mat(angles):
    """
    3x3 transformation matrix for rotation along x, y, z axes.
    """
    x_mat = rot_mat_x_axis(angles[0])
    y_mat = rot_mat_y_axis(angles[1])
    z_mat = rot_mat_z_axis(angles[2])
    return z_mat @ y_mat @ x_mat

def trans_mat(offsets):
    """
    4x4 transformation matrix for translation along x, y, z axes.
    """
    res = np.identity(4, dtype = np.float32)
    res[:3, 3] = np.array(offsets)
    return res

def rot_trans_mat(offsets, angles):
    """
    4x4 transformation matrix for rotation along x, y, z axes, then translation along x, y, z axes.
    """
    res = np.identity(4, dtype = np.float32)
    res[:3, :3] = rot_mat(angles)
    res[:3, 3] = np.array(offsets)
    return res

def trans_rot_mat(offsets, angles):
    """
    4x4 transformation matrix for translation along x, y, z axes, then rotation along x, y, z axes.
    """
    res = np.identity(4, dtype = np.float32)
    res[:3, :3] = rot_mat(angles)
    offsets = res[:3, :3] @ np.asarray(offsets, dtype=np.float32)
    res[:3, 3] = offsets
    return res
