import numpy as np
import asyncio
import json
import websockets
import torch
import cv2

import msgpack
import msgpack_numpy as m
m.patch()

from polaris.client.abstract_client import InferenceClient, PolicyArgs
from polaris.utils_.planner_utils import setup_curobo
import time


@InferenceClient.register(client_name="mimicplay")
class MimicPlayClient(InferenceClient):

    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        self.uri = f"ws://{args.host}:{args.port}"
        self.open_loop_horizon = args.open_loop_horizon
        self.pred_action_chunk = None
        self.motion_gen = setup_curobo()
        self.device = args.device

    def delta_to_absolute(self, action_delta: np.ndarray, current_obs: dict,
                          max_dpos: float = 0.05, max_drot: float = 0.5) -> np.ndarray:
        """
        Inverse of compute_delta_actions_robomimic.

        Args:
            action_delta:  (7,) = delta_pos(3) + delta_axis_angle(3) + gripper(1), normalized
            current_obs:   raw obs dict; current_obs["policy"]["ee_pose"] is (1, 7) = pos(3) + quat_wxyz(4)
            max_dpos:      normalization scale for position delta
            max_drot:      normalization scale for rotation delta

        Returns:
            (8,) = pos(3) + quat_wxyz(4) + gripper(1)  with gripper in [0, 1]
        """
        from scipy.spatial.transform import Rotation as R

        ee_pose = current_obs["policy"]["ee_pose"][0]
        if isinstance(ee_pose, torch.Tensor):
            ee_pose = ee_pose.cpu().numpy()

        curr_pos = ee_pose[:3]
        curr_quat_wxyz = ee_pose[3:7]

        # Unnormalize deltas
        delta_pos = action_delta[:3] * max_dpos
        delta_aa  = action_delta[3:6] * max_drot
        gripper   = action_delta[6]

        # Target position
        target_pos = curr_pos + delta_pos

        # Current rotation (scipy uses xyzw)
        curr_quat_xyzw = curr_quat_wxyz[[1, 2, 3, 0]]
        curr_rot = R.from_quat(curr_quat_xyzw)

        # Delta rotation from axis-angle, then compose: target = delta @ curr
        delta_rot  = R.from_rotvec(delta_aa)
        target_rot = delta_rot * curr_rot

        # Back to wxyz
        xyzw = target_rot.as_quat()
        target_quat_wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

        # Gripper: [-1, +1] -> [0, 1]
        gripper_abs = (gripper + 1.0) / 2.0

        result = np.concatenate([target_pos, target_quat_wxyz, [gripper_abs]]).astype(np.float32)
        return torch.from_numpy(result).to(self.device)

    def ee_to_joint(self, action_ee):
        ee_pos = action_ee[:3]
        ee_quat = action_ee[3:7] # wxyz
        gripper = action_ee[-1]

        ee_pos = ee_pos.to(self.device)
        ee_quat = ee_quat.to(self.device)

        from curobo.types.math import Pose

        goal_pose = Pose(position=ee_pos, quaternion=ee_quat)
        ik_result = self.motion_gen.ik_solver.solve_single(goal_pose)
        joint_solution = ik_result.solution.squeeze(0)[:, :8]
        joint_solution[:, -1] = gripper
        
        return joint_solution # (1, 8)

    def reset(self):
        self.pred_action_chunk = None
        asyncio.run(self._send({
            "command": "reset",
        }))

    def _serialize(self, v, _path=""):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().numpy()
        if isinstance(v, dict):
            return {k: self._serialize(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [self._serialize(i) for i in v]
        return v

    async def _send(self, message: dict) -> dict:
        async with websockets.connect(self.uri, max_size=100 * 1024 * 1024) as ws:
            packed = msgpack.packb(message, use_bin_type=True)
            await ws.send(packed)
            response = await ws.recv()
            return msgpack.unpackb(response, raw=False)

    def infer(
        self, obs: dict, language_instruction=None, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        
        result = asyncio.run(self._send({
            "command": "infer",
            "obs": self._serialize(self._extract_observation(obs)),
            "return_viz": return_viz,
        }))

        delta_action_ee = np.array(result["action"])
        action_ee = self.delta_to_absolute(delta_action_ee, obs)
        
        action = self.ee_to_joint(action_ee)

        action = action.detach().cpu().numpy()
        self._viz = np.array(result["viz"]) if result.get("viz") is not None else None

        return action, self._viz if return_viz else None

    def apply_image_preprocess(self, img: np.ndarray, shift: int = 100, img_size: int = 84,
                            original_w: int = 1280, original_h: int = 720) -> np.ndarray:
        """
        Preprocess image: center-crop with pixel shift, resize, normalize.

        Args:
            img:        (H, W, C) uint8 RGB image
            shift:      horizontal pixel shift for crop (PIXEL_SHIFT)
            img_size:   target square size after resize
            original_w: original image width
            original_h: original image height

        Returns:
            (C, H, W) float32 numpy array normalized to [0, 1]
        """
        side = min(original_h, original_w)
        crop_x1 = (original_w - side) // 2 + shift
        crop_y1 = (original_h - side) // 2
        crop_x2 = crop_x1 + side
        crop_y2 = crop_y1 + side

        # Crop
        img = img[crop_y1:crop_y2, crop_x1:crop_x2]

        # Resize
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        # HWC -> CHW, normalize
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

        return img

    
    def _extract_observation(self, obs_dict):
        # Assign images
        agentview_image = obs_dict["splat"]["cam1"]
        robot0_eye_in_hand_image = obs_dict["splat"]["wrist_cam"]

        agentview_image = self.apply_image_preprocess(agentview_image)
        robot0_eye_in_hand_image = self.apply_image_preprocess(robot0_eye_in_hand_image)

        robot_state = obs_dict["policy"]
        robot0_eef_pos = robot_state["ee_pose"][:, :3]
        robot0_eef_quat = robot_state["ee_pose"][:, 3:7][:, [1, 2, 3, 0]] # wxyz -> xyzw

        gripper = robot_state["gripper_width"]
        robot0_gripper_qpos = torch.cat([gripper, -gripper], axis=-1)
        return {
            "agentview_image": torch.from_numpy(agentview_image).unsqueeze(0).to(self.device), # (1, C, H, W)
            "robot0_eye_in_hand_image": torch.from_numpy(robot0_eye_in_hand_image).unsqueeze(0).to(self.device),
            "robot0_eef_pos": robot0_eef_pos.to(self.device),
            "robot0_eef_quat": robot0_eef_quat.to(self.device),
            "robot0_gripper_qpos": robot0_gripper_qpos.to(self.device),
        }
    
    def shutdown(self):
        """Send shutdown signal to THIS server (by port)."""
        try:
            asyncio.run(self._send({"command": "shutdown"}))
            print(f"Shutdown signal sent to {self.uri}")
        except Exception as e:
            print(f"Server {self.uri} closed: {e}")
