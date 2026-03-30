import numpy as np
import os
import json
import imageio.v3 as iio

from polaris.utils_.vis_utils import debug_plot

# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

class ObsRecorder:
    def __init__(self, calibration_path: str, save_dir: str, fps: int = 30, ep_idx: int = 0):

        self.save_dir = save_dir
        self.ep_idx = ep_idx
        self.fps = fps

        self.next_event_idx = []
        self.all_obs = []
        self.calibration = self.get_cam_param(calibration_path)

        self.debug_frames = []
        self.debug = True

    def get_cam_param(self, calibration_path):
        with open(calibration_path, "r") as f:
            cams = json.load(f)

        calibration = {}
        for name, c in cams.items():
            calibration[name] = {
                "intrinsic":  np.array(c["intrinsic"]),
                "extrinsic":  np.array(c["extrinsic"]),  # cam_to_base (4, 4)
                "distortion": np.array(c["distortion"]),
            }

        return calibration

    def add(self, obs):
  
        self.all_obs.append(obs)

    def save_subgoal(self):
        self.next_event_idx.append(len(self.all_obs) - 1)
        print(f"  Subgoal reached at frame {len(self.all_obs) - 1}")

    def process_obs(self):
        if not self.next_event_idx:
            print("[warn] no subgoals recorded, skipping goal_gripper_pcd")
            return
        
        if len(self.all_obs) == 1:
            if self.debug:
                self.debug_frames.append(debug_plot(self.all_obs[0]["splat"]["cam1"], 
                                                    self.all_obs[0]["policy"]["gripper_pcd"], 
                                                    self.calibration["cam1"]["intrinsic"], 
                                                    self.calibration["cam1"]["extrinsic"],
                                                    self.all_obs[0]["policy"]["ee_pose"]))

            return 
        

        for idx in range(len(self.all_obs)): 
            
            if self.debug:
                self.debug_frames.append(debug_plot(self.all_obs[idx]["splat"]["cam1"], 
                                                    self.all_obs[idx]["policy"]["gripper_pcd"], 
                                                    self.calibration["cam1"]["intrinsic"], 
                                                    self.calibration["cam1"]["extrinsic"],
                                                    self.all_obs[idx]["policy"]["ee_pose"]))

            # find the next event index >= idx
            next_event_idx = next(
                (self.next_event_idx[i] for i in range(len(self.next_event_idx))
                if self.next_event_idx[i] > idx),
                self.next_event_idx[-1]
            )
            #self.all_obs[idx]["goal_gripper_pcd"] = self.all_obs[next_event_idx]["gripper_pcd"]


    def save_episode(self):
        self.process_obs()
        ep_dir = os.path.join(self.save_dir, f"episode_{self.ep_idx:06d}")
        os.makedirs(ep_dir, exist_ok=True)

        # Collect arrays from all_obs (skip first obs which has no action)
        all_obs = self.all_obs
        # N = T+1 states (includes terminal state with no action)
        N = len(all_obs)

        for cam in self.calibration.keys():
            frames  = np.stack([o["splat"][cam] for o in all_obs if o.get("splat") is not None])
            iio.imwrite(os.path.join(ep_dir, f"{cam}.mp4"), frames, fps=self.fps)

        if self.debug_frames:
            debug_frames = np.stack(self.debug_frames) 
            iio.imwrite(os.path.join(ep_dir, "debug_video.mp4"), debug_frames, fps=self.fps)

        wrist_frames  = np.stack([o["splat"]["wrist_cam"] for o in all_obs]) # T
        # states        = np.stack([o["state"]       for o in all_obs]) # (T, 8)
        # abs_actions   = np.stack([o["abs_action"]  for o in all_obs if o["abs_action"] is not None]) # (T-1, 8)
        # gripper_pcd  = np.stack([o["gripper_pcd"] for o in all_obs]) # (T, 4, 3)
        # goal_gripper_pcd     = np.stack([o["goal_gripper_pcd"] for o in all_obs]) # (T, 4, 3)
        # object_pcd    = np.stack([o["object_pcd"] for o in all_obs]) # (T, 4500, 3)

        iio.imwrite(os.path.join(ep_dir, "wrist_cam.mp4"), wrist_frames, fps=self.fps)
        # np.savez(
        #     os.path.join(ep_dir, "trajectory.npz"),
        #     # states        = states.astype(np.float32),
        #     # abs_actions   = abs_actions.astype(np.float32),
        #     # gripper_pcd  = gripper_pcd.astype(np.float32),
        #     # goal_gripper_pcd     = goal_gripper_pcd.astype(np.float32),
        #     # object_pcd    = object_pcd.astype(np.float32),
        # )

        print(f"  [saved] {ep_dir}/")
        # print(f"           states       : {states.shape}")
        # print(f"           abs_actions  : {abs_actions.shape}")
        # print(f"           gripper_pcds : {gripper_pcd.shape}")
        # print(f"           goal_gripper_pcds    : {goal_gripper_pcd.shape}")
        # print(f"           object_pcd   : {object_pcd.shape}")


    def reset(self):
        self.next_event_idx = []
        self.all_obs = []

        self.debug_frames = []