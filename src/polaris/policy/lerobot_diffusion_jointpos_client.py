import numpy as np
import torch
from collections import deque
from lerobot.common.policies.pretrained import PreTrainedConfig
from lerobot.common.policies.factory import get_policy_class

from polaris.policy.abstract_client import InferenceClient
from polaris.config import PolicyArgs
import os


@InferenceClient.register(client_name="LeRobotDiffusion")
class LeRobotDiffusionClient(InferenceClient):

    N_OBS_STEPS = 2
    IMAGE_H = 720
    IMAGE_W = 1280

    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        if args.policy_path is None:
            raise ValueError("policy_path must be set for LeRobotDiffusionClient")
        if args.open_loop_horizon is None:
            raise ValueError("open_loop_horizon must be set for LeRobotDiffusionClient")
        
        self.policy_config: PreTrainedConfig = PreTrainedConfig.from_pretrained(args.policy_path)
        self.policy_config.pretrained_path = args.policy_path
        import lerobot
        lerobot_dir = os.path.dirname(os.path.dirname(lerobot.__file__))
        self.policy_config.calibration_json = os.path.join(lerobot_dir, self.policy_config.calibration_json)

        policy_cls = get_policy_class(self.policy_config.type)
        kwargs = {}
        kwargs["config"] = self.policy_config
        kwargs["pretrained_name_or_path"] = args.policy_path

        self.policy = policy_cls.from_pretrained(**kwargs)

        self.policy.eval()
        self.policy.to("cuda")

        self.open_loop_horizon = args.open_loop_horizon
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

        # Rolling buffers for n_obs_steps=2
        self.exterior_buf = deque(maxlen=self.N_OBS_STEPS)
        self.wrist_buf    = deque(maxlen=self.N_OBS_STEPS)
        self.state_buf    = deque(maxlen=self.N_OBS_STEPS)

    @property
    def rerender(self) -> bool:
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.exterior_buf.clear()
        self.wrist_buf.clear()
        self.state_buf.clear()
        self.policy.reset()

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:

        curr_obs = self._extract_observation(obs)

        # Always push latest obs into rolling buffers
        self.exterior_buf.append(curr_obs["exterior_image"])  # (H, W, 3) uint8
        self.wrist_buf.append(curr_obs["wrist_image"])
        self.state_buf.append(curr_obs["state"])

        viz = None
        if return_viz:
            import cv2
            ext_small  = cv2.resize(curr_obs["exterior_image"], (224, 224))
            wrist_small = cv2.resize(curr_obs["wrist_image"],   (224, 224))
            viz = np.concatenate([ext_small, wrist_small], axis=1)

        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            self.actions_from_chunk_completed = 0
            lerobot_obs = self._build_lerobot_obs(instruction)

            with torch.inference_mode():
                # select_action returns (action_dim,) for one step,
                # called repeatedly internally by diffusion policy
                action_tensor, action_eef = self.policy.select_action(lerobot_obs)


            # select_action returns a single action per call in LeRobot diffusion;
            # it internally manages the chunk and serves one action at a time.
            # So we just store the single action and reset the counter to 1.
            self.pred_action_chunk = action_tensor.squeeze(0).cpu().numpy()  # (action_dim,)

            self.actions_from_chunk_completed = 1

            action = self.pred_action_chunk
        else:
            # Policy is stateful between select_action calls — just query again
            lerobot_obs = self._build_lerobot_obs(instruction)
            with torch.inference_mode():
                action_tensor, action_eef = self.policy.select_action(lerobot_obs)
            action = action_tensor.squeeze(0).cpu().numpy()
            self.actions_from_chunk_completed += 1

        # Binarize gripper (last dim)
        action = np.concatenate([
            action[:-1],
            np.ones((1,)) if action[-1] > 0.5 else np.zeros((1,))
        ])

        return action, viz

    def _build_lerobot_obs(self, instruction: str) -> dict:
        def img_to_tensor(img: np.ndarray) -> torch.Tensor:
            # img: (H, W, 3) uint8
            t = torch.from_numpy(img).float() / 255.0  # (H, W, 3)
            t = t.permute(2, 0, 1)                     # (3, H, W)
            return t.unsqueeze(0).to("cuda")           # (1, 3, H, W)

        state_tensor = torch.from_numpy(self.state_buf[-1]).float().unsqueeze(0).to("cuda")  # (1, 8)

        return {
            "observation.images.cam_azure_kinect_front.color": img_to_tensor(self.exterior_buf[-1]),
            "observation.images.cam_wrist":                    img_to_tensor(self.wrist_buf[-1]),
            "observation.state":                               state_tensor,
            "task":                                            [instruction],
        }

    def _extract_observation(self, obs_dict: dict) -> dict:
        exterior_image = obs_dict["splat"]["external_cam"]   # (H, W, 3) uint8
        wrist_image    = obs_dict["splat"]["wrist_cam"]

        robot_state    = obs_dict["policy"]
        joint_pos      = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()[0]  # (7,)
        gripper_pos    = robot_state["gripper_pos"].clone().detach().cpu().numpy()[0]    # (1,)
        state          = np.concatenate([joint_pos, gripper_pos], axis=0)               # (8,)

        return {
            "exterior_image": exterior_image,
            "wrist_image":    wrist_image,
            "state":          state,
        }