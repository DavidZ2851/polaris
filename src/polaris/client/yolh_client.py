import pickle
from collections import deque
from typing import Optional, Tuple

import numpy as np
import zmq
from scipy.spatial.transform import Rotation
import torch

from polaris.client.abstract_client import InferenceClient
from polaris.config import PolicyArgs
# from polaris.policy.yolh.utils.transformation import rot6d_to_matrix
from polaris.utils_.planner_utils import setup_curobo

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

@InferenceClient.register(client_name="YOLH")
class YOLHClient(InferenceClient):
    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        host = args.host if args.host is not None else "localhost"
        port = args.port if args.port is not None else 5557
        self.cam_key = args.cam_key if args.cam_key is not None else "cam1"
        self.obs_horizon = max(1, getattr(args, "obs_horizon", 2) or 2)

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"Connected to YOLH server at {host}:{port}")

        self.motion_gen = setup_curobo()
        self.action_chunk: Optional[np.ndarray] = None
        self.actions_from_chunk_completed = 0
        self.current_chunk_horizon = 0
        self.ee_pose_history: deque[np.ndarray] = deque(maxlen=self.obs_horizon)

    @property
    def rerender(self) -> bool:
        return (
            self.action_chunk is None
            or self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.current_chunk_horizon
        )

    def reset(self):
        self.action_chunk = None
        self.actions_from_chunk_completed = 0
        self.current_chunk_horizon = 0
        self.ee_pose_history.clear()
        self.socket.send(pickle.dumps({"reset": True}))
        self.socket.recv()

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        self._update_ee_pose_history(obs)

        if self._need_new_chunk():
            request = self._build_request(obs)
            self.socket.send(pickle.dumps(request))
            response = pickle.loads(self.socket.recv())
            self.action_chunk = np.asarray(response["action_chunk"], dtype=np.float32)
            self.actions_from_chunk_completed = 0
            self.current_chunk_horizon = self._compute_chunk_horizon()

        if self.action_chunk is None or len(self.action_chunk) == 0:
            action = self._hold_current_action(obs)
        else:
            action10 = self.action_chunk[self.actions_from_chunk_completed]
            self.actions_from_chunk_completed += 1
            action = self._policy_action_to_env_action(action10, obs)

        viz = obs["splat"][self.cam_key] if return_viz else None
        return action, viz

    def _need_new_chunk(self) -> bool:
        return (
            self.action_chunk is None
            or len(self.action_chunk) == 0
            or self.actions_from_chunk_completed >= self.current_chunk_horizon
        )

    def _compute_chunk_horizon(self) -> int:
        if self.action_chunk is None:
            return 0
        chunk_len = len(self.action_chunk)
        if self.args.open_loop_horizon is None:
            return chunk_len
        return min(chunk_len, self.args.open_loop_horizon)

    def _build_request(self, obs_dict: dict) -> dict:
        rgb = np.asarray(obs_dict["splat"][self.cam_key])
        depth = np.asarray(obs_dict["splat"][f"{self.cam_key}_depth"])
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]

        request = {
            "rgb": rgb,
            "depth": depth,
            "gripper_pcd": obs_dict["policy"]["gripper_pcd"][0].detach().cpu().numpy(),
            "ee_pose_history": self._get_padded_ee_pose_history(),
        }

        mask_key = f"{self.cam_key}_mask"
        if mask_key in obs_dict["splat"]:
            request["robot_mask"] = np.asarray(obs_dict["splat"][mask_key]).astype(np.bool_)

        return request

    def _update_ee_pose_history(self, obs_dict: dict) -> None:
        ee_pose = obs_dict["policy"]["ee_pose"][0].detach().cpu().numpy().astype(np.float32)
        self.ee_pose_history.append(ee_pose)

    def _get_padded_ee_pose_history(self) -> np.ndarray:
        if not self.ee_pose_history:
            raise RuntimeError("ee_pose_history is empty; call infer with a valid observation first")

        history = list(self.ee_pose_history)
        if len(history) < self.obs_horizon:
            history = [history[0].copy() for _ in range(self.obs_horizon - len(history))] + history
        return np.stack(history[-self.obs_horizon :]).astype(np.float32)

    def _hold_current_action(self, obs: dict) -> np.ndarray:
        joint_pos = obs["policy"]["arm_joint_pos"][0].detach().cpu().numpy()
        gripper = obs["policy"]["gripper_pos"][0].detach().cpu().numpy()
        gripper_bin = np.array([1.0 if gripper[0] > 0.5 else 0.0], dtype=np.float32)
        return np.concatenate([joint_pos, gripper_bin]).astype(np.float32)

    def _policy_action_to_env_action(self, action: np.ndarray, obs: dict) -> np.ndarray:
        from curobo.types.math import Pose

        pos = action[:3]
        rot6d = action[3:9]
        gripper = action[9]

        rot_matrix = rot6d_to_matrix(rot6d)
        quat_xyzw = Rotation.from_matrix(rot_matrix).as_quat()
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float32,
        )

        goal_pose = Pose(
            position=torch.from_numpy(pos).float().unsqueeze(0).to(self.args.device),
            quaternion=torch.from_numpy(quat_wxyz).float().unsqueeze(0).to(self.args.device),
        )
        ik_result = self.motion_gen.ik_solver.solve_single(goal_pose)

        if ik_result.success.item():
            joint_pos = ik_result.solution.squeeze(0)[0, :7].cpu().numpy()
        else:
            print("[YOLHClient] IK failed, holding current joints")
            joint_pos = obs["policy"]["arm_joint_pos"][0].detach().cpu().numpy()

        gripper_bin = np.array([1.0 if gripper > 0.5 else 0.0], dtype=np.float32)
        return np.concatenate([joint_pos, gripper_bin]).astype(np.float32)