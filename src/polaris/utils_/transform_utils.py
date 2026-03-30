import numpy as np
import robosuite.utils.transform_utils as T

MAX_GRIPPER_WIDTH = 0.05

def rotation_transfer_6D_to_matrix(orient):
    if type(orient) == list or type(orient) == tuple:
        orient = np.array(orient, dtype=np.float64)

    orient = orient.reshape(2, 3)
    a1 = orient[0]
    a2 = orient[1]

    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(a2, b1) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)

    rotate_matrix = np.array([b1, b2, b3], dtype=np.float64).T

    return rotate_matrix

def rotation_transfer_matrix_to_6D(rotate_matrix):
    if type(rotate_matrix) == list or type(rotate_matrix) == tuple:
        rotate_matrix = np.array(rotate_matrix, dtype=np.float64).reshape(3, 3)
    rotate_matrix = rotate_matrix.reshape(3, 3)
    
    a1 = rotate_matrix[:, 0]
    a2 = rotate_matrix[:, 1]

    orient = np.array([a1, a2], dtype=np.float64).flatten()
    return orient

def quat_wxyz_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (wxyz) to 6D rotation representation.
    
    Args:
        quat_wxyz (4,)  quaternion in wxyz format
    
    Returns:
        rot6d (6,)  first two columns of rotation matrix, flattened
    """
    # wxyz → xyzw for T.quat2mat
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    R = T.quat2mat(quat_xyzw)  # (3, 3)
    return rotation_transfer_matrix_to_6D(R).astype(np.float32)  # (6,)

def state_quat_to_rot6d(state: np.ndarray) -> np.ndarray:
    """
    Convert state from quat to rot6d representation.
    
    Args:
        state (8,)   [pos(3) | quat_wxyz(4) | gripper(1)]
    
    Returns:
        state (10,)  [pos(3) | rot6d(6) | gripper(1)]
    """
    pos     = state[:3]
    rot6d   = quat_wxyz_to_rot6d(state[3:7])
    gripper = state[7:8]
    return np.concatenate([pos, rot6d, gripper]).astype(np.float32)  # (10,)

def clip_delta_action(delta_pos: np.array, delta_local_rot: np.array, max_dpos: float = 0.05, max_drot: float = 0.5):
    delta_pos_clipped = np.clip(delta_pos, -max_dpos, max_dpos)
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(delta_local_rot)
    rotvec = r.as_rotvec()                         # (3,) axis-angle
    angle  = np.linalg.norm(rotvec)
    if angle > max_drot:
        rotvec = rotvec / angle * max_drot         # scale down to max_drot
    delta_local_rot_clipped = Rotation.from_rotvec(rotvec).as_matrix()

    return delta_pos_clipped, delta_local_rot_clipped

def compute_delta_action(state: np.ndarray, target_pos: np.ndarray, clip_action: bool = False) -> np.ndarray:
    """
    Args:
        state      (8,)   [pos(3) | quat_wxyz(4) | gripper(1)]
        abs_action (8,)   [pos(3) | quat_wxyz(4) | gripper(1)]

    Returns:
        delta (10,)  [delta_pos(3) | delta_rot6d(6) | gripper(1) normalized (-0.01, 0.01)]
    """
    # --- delta pos (base frame) ---
    delta_pos = target_pos[:3] - state[:3]

    # --- delta rot (EEF local frame) ---
    # state quat is wxyz → convert to xyzw for T.quat2mat
   
    cur_quat_xyzw    = np.array([state[4],      state[5],      state[6],      state[3]])
    action_quat_xyzw = np.array([target_pos[4], target_pos[5], target_pos[6], target_pos[3]])

    cur_rot    = T.quat2mat(cur_quat_xyzw)       # (3, 3) 
    action_rot = T.quat2mat(action_quat_xyzw)    # (3, 3) 

    # local frame: cur_rot = delta_local_rot @ target_pos
    delta_local_rot = cur_rot.T @ action_rot
    
    if clip_action:
        delta_pos, delta_local_rot = clip_delta_action(delta_pos, delta_local_rot)
    delta_rot_6d = rotation_transfer_matrix_to_6D(delta_local_rot).reshape(-1).astype(np.float32)

    # --- normalized gripper to SMITH dataset ---
    gripper_normalized = -0.01 if target_pos[-1] > MAX_GRIPPER_WIDTH / 2 else 0.01

    return np.concatenate([delta_pos, delta_rot_6d, [gripper_normalized]]).astype(np.float32)  # (10,)
