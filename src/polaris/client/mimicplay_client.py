import numpy as np
import asyncio
import json
import websockets
import torch

import msgpack
import msgpack_numpy as m
m.patch()

from polaris.policy.abstract_client import InferenceClient, PolicyArgs
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
    
    def ee_to_joint(self, action_ee):
        ee_pos = action_ee[:3]
        ee_quat = action_ee[3:7] # wxyz
        gripper = action_ee[-1]

        ee_pos = torch.from_numpy(ee_pos).to(self.device)
        ee_quat = torch.from_numpy(ee_quat).to(self.device)

        from curobo.types.math import Pose

        goal_pose = Pose(position=ee_pos, quaternion=ee_quat)
        ik_result = self.motion_gen.ik_solver.solve_single(goal_pose)
        joint_solution = ik_result.solution.squeeze(0)[:, :8]
        joint_solution[:, -1] = gripper
        
        return joint_solution # (1, 8)

    def reset(self, obs):
        self.pred_action_chunk = None
        asyncio.run(self._send({
            "command": "reset",
            "obs": self._serialize(self._extract_observation(obs)),
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

        action_ee = np.array(result["action"])
        
        action = self.ee_to_joint(action_ee)

        action = action.detach().cpu().numpy()
        self._viz = np.array(result["viz"]) if result.get("viz") is not None else None

        return action, self._viz if return_viz else None
    
    def _extract_observation(self, obs_dict):
        # Assign images
        agentview_image = obs_dict["splat"]["cam1"]
        robot0_eye_in_hand_image = obs_dict["splat"]["wrist_cam"]

        robot_state = obs_dict["policy"]["ee_pose"]
        
        robot0_eef_pos = robot_state["ee_pose"][:, :3]
        robot0_eef_quat = robot_state["ee_pose"][:, 3:7][:, 1, 2, 3, 0] # wxyz -> xyzw

        gripper = robot_state["gripper_width"]
        robot0_gripper_qpos = np.concatenate([gripper, -gripper], axis=-1)


        return {
            "agentview_image": agentview_image,
            "robot0_eye_in_hand_image": robot0_eye_in_hand_image,
            "robot0_eef_pos": robot0_eef_pos,
            "robot0_eef_quat": robot0_eef_quat,
            "robot0_gripper_qpos": robot0_gripper_qpos,
        }
    
    def shutdown(self):
        """Send shutdown signal to THIS server (by port)."""
        try:
            asyncio.run(self._send({"command": "shutdown"}))
            print(f"Shutdown signal sent to {self.uri}")
        except Exception as e:
            print(f"Server {self.uri} closed: {e}")
