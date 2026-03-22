import os
import tyro
import argparse
import json

import torch
import numpy as np
import imageio.v3 as iio
import gymnasium as gym


from isaaclab.app import AppLauncher
from polaris.config import DataArgs
from scipy.spatial.transform import Rotation

ISAAC_TO_CUROBO_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 8, 10, 12]

def debug_plot(frame, pcd, K, T_cam_to_base):
    import cv2

    img = frame.copy()
    T_base_to_cam = np.linalg.inv(T_cam_to_base)

    def project(pts_world):
        pts_h   = np.concatenate([pts_world, np.ones((len(pts_world), 1))], axis=1)
        pts_cam = (T_base_to_cam @ pts_h.T).T[:, :3]
        pts2d   = (K @ pts_cam.T).T
        pts2d   = (pts2d[:, :2] / pts2d[:, 2:3]).astype(int)
        return pts2d

    # ── Gripper PCD ───────────────────────────────────────────────────────────
    pts2d  = project(pcd)
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0)]
    labels = ["grasp", "left", "right", "top"]

    for pt, color, label in zip(pts2d, colors, labels):
        if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
            cv2.circle(img, tuple(pt), 5, color, -1)
            cv2.putText(img, label, tuple(pt + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if all(0 <= pts2d[i][0] < img.shape[1] and 0 <= pts2d[i][1] < img.shape[0] for i in [1, 2]):
        cv2.line(img, tuple(pts2d[1]), tuple(pts2d[2]), (255,255,255), 1)

    # ── World origin axes ─────────────────────────────────────────────────────
    axis_len = 0.1
    origins = np.array([
        [0.0, 0.0, 0.0],
        [axis_len, 0.0, 0.0],  # X
        [0.0, axis_len, 0.0],  # Y
        [0.0, 0.0, axis_len],  # Z
    ])
    apts = project(origins)
    o = tuple(apts[0])
    ax_colors = [(0,0,255), (0,255,0), (255,0,0)]  # X=red, Y=green, Z=blue
    ax_labels = ["X", "Y", "Z"]
    for j, (ac, al) in enumerate(zip(ax_colors, ax_labels)):
        ep = tuple(apts[j+1])
        h, w = img.shape[:2]
        if all(0 <= p[0] < w and 0 <= p[1] < h for p in [o, ep]):
            cv2.arrowedLine(img, o, ep, ac, 2, tipLength=0.2)
            cv2.putText(img, al, ep, cv2.FONT_HERSHEY_SIMPLEX, 0.4, ac, 1)

    return img


def main(data_args: DataArgs):
    parser = argparse.ArgumentParser()
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = data_args.headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import polaris.environments
    from isaaclab_tasks.utils import parse_env_cfg
    from polaris.environments.manager_based_rl_splat_environment import ManagerBasedRLSplatEnv
    from polaris.utils import load_eval_initial_conditions, load_task_config
    
    if data_args.env_folder is None:
        data_args.env_folder = os.path.dirname(gym.spec(data_args.env).kwargs["usd_file"])
 
    ROBOT_CONFIG = f"{data_args.robot}.yml"
    DEVICE       = data_args.device
    env_name     = data_args.env
    env_folder   = data_args.env_folder
 
    env_cfg = parse_env_cfg(env_name, device=DEVICE, num_envs=1, use_fabric=True)
    env: ManagerBasedRLSplatEnv = gym.make(env_name, cfg=env_cfg)
 
    language_instruction, initial_conditions = load_eval_initial_conditions(env.usd_file)
    object_randomization, waypoints = load_task_config(os.path.join(env_folder, "task_config.yaml"))
 
    object_initialization = randomize_object_poses(object_randomization, initial_conditions)
    obs, info = env.reset(object_positions=object_initialization[0])

    print("Task:", language_instruction)
 
    recorder = ObsRecorder("/home/haotian/polaris/PolaRiS-Hub/put_red_cup_no_curtain/cam_calibration.json", "/home/haotian/polaris/debug", fps=30)
 
    planner = MotionPlanner(
        env=env,
        recorder=recorder,
        robot_config=ROBOT_CONFIG,
        waypoints=waypoints,
        steps_per_waypoint=data_args.steps_per_waypoint,
        device=DEVICE,
    )

    planner.add_obs(obs)
 
    print("cuRobo MotionGen ready.\n")
    print("play trajectories")
 
    obs, info, done = planner.execute_waypoints(
        obs,
        object_poses=object_initialization[0],
    )
 
    env.close()
    simulation_app.close()


class ObsRecorder:
    def __init__(self, calibration_path: str, save_dir: str, fps: int = 30, ep_idx: int = 0):

        self.save_dir = save_dir
        self.ep_idx = ep_idx
        self.fps = fps

        self.object_pcd = None
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
        
        if obs["object_pcd"] is not None and self.object_pcd is None:
            self.object_pcd = obs["object_pcd"]

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
                                                    self.all_obs[0]["gripper_pcd"], 
                                                    self.calibration["cam1"]["intrinsic"], 
                                                    self.calibration["cam1"]["extrinsic"]))

            return 
        
        previous_action = None

        for idx in range(len(self.all_obs)): 
            
            if self.debug:
                self.debug_frames.append(debug_plot(self.all_obs[idx]["splat"]["cam1"], 
                                                    self.all_obs[idx]["gripper_pcd"], 
                                                    self.calibration["cam1"]["intrinsic"], 
                                                    self.calibration["cam1"]["extrinsic"]))

            abs_action = self.all_obs[idx]["abs_action"]

            if previous_action is None:
                delta_action = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=np.float32)
            else:
                delta_action = self.compute_delta_action(abs_action, previous_action)

            self.all_obs[idx]["del_action"] = delta_action
            previous_action = abs_action

            # find the next event index >= idx
            next_event_idx = next(
                (self.next_event_idx[i] for i in range(len(self.next_event_idx))
                if self.next_event_idx[i] > idx),
                self.next_event_idx[-1]
            )
            self.all_obs[idx]["goal_gripper_pcd"] = self.all_obs[next_event_idx]["gripper_pcd"]


    def compute_delta_action(self, current_action, previous_action):
        """
        current_action:  10-dim [pos(3) | rot6d(6) | width(1)]
        previous_action: 10-dim [pos(3) | rot6d(6) | width(1)]
        Both in numpy.
        """
        from scipy.spatial.transform import Rotation
        import numpy as np

        # --- delta pos (in base frame) ---
        delta_pos = current_action[:3] - previous_action[:3]

        # --- delta rot (in eef frame) ---
        def rot6d_to_matrix(r):
            a1, a2 = r[:3], r[3:]
            b1 = a1 / np.linalg.norm(a1)
            b2 = a2 - np.dot(b1, a2) * b1
            b2 = b2 / np.linalg.norm(b2)
            b3 = np.cross(b1, b2)
            return np.stack([b1, b2, b3], axis=-1)  # (3, 3)

        def matrix_to_rot6d(R):
            return R[:, :2].T.flatten()  # first two columns, flattened

        R_cur  = rot6d_to_matrix(current_action[3:9])
        R_prev = rot6d_to_matrix(previous_action[3:9])

        # delta rotation in eef frame: R_prev^T @ R_cur
        R_delta = R_prev.T @ R_cur
        delta_rot = matrix_to_rot6d(R_delta)

        # --- delta width ---
        delta_width = current_action[9:10] - previous_action[9:10]

        return np.concatenate([delta_pos, delta_rot, delta_width])  # (10,)

    def save_episode(self):
        self.process_obs()
        ep_dir = os.path.join(self.save_dir, f"episode_{self.ep_idx:06d}")
        os.makedirs(ep_dir, exist_ok=True)

        # Collect arrays from all_obs (skip first obs which has no action)
        all_obs = self.all_obs
        N = len(all_obs)

        for cam in self.calibration.keys():
            frames  = np.stack([o["splat"][cam] for o in all_obs if o.get("splat") is not None])
            iio.imwrite(os.path.join(ep_dir, f"{cam}.mp4"), frames, fps=self.fps)

        if self.debug_frames:
            debug_frames = np.stack(self.debug_frames) 
            iio.imwrite(os.path.join(ep_dir, "debug_video.mp4"), debug_frames, fps=self.fps)

        wrist_frames  = np.stack([o["splat"]["wrist_cam"] for o in all_obs if o.get("splat") is not None])
        states        = np.stack([o["state"]       for o in all_obs if o.get("state") is not None])
        abs_actions   = np.stack([o["abs_action"]  for o in all_obs if o.get("abs_action") is not None])
        del_actions   = np.stack([o["del_action"]  for o in all_obs if o.get("del_action") is not None])
        gripper_pcds  = np.stack([o["gripper_pcd"] for o in all_obs if o.get("gripper_pcd") is not None])
        goal_pcds     = np.stack([o["goal_gripper_pcd"] for o in all_obs if o.get("goal_gripper_pcd") is not None])
        object_pcd    = self.object_pcd.astype(np.float32)

        iio.imwrite(os.path.join(ep_dir, "wrist_cam.mp4"), wrist_frames, fps=self.fps)
        np.savez(
            os.path.join(ep_dir, "trajectory.npz"),
            states        = states.astype(np.float32),
            abs_actions   = abs_actions.astype(np.float32),
            del_actions   = del_actions.astype(np.float32),
            gripper_pcds  = gripper_pcds.astype(np.float32),
            goal_pcds     = goal_pcds.astype(np.float32),
            object_pcd    = object_pcd.astype(np.float32),
        )

        print(f"  [saved] {ep_dir}/")
        print(f"           ├── video.mp4  ({len(self.all_obs)} frames)")
        print(f"           └── trajectory.npz  ({len(states)} steps)")


    def reset(self):
        self.object_pcd = None
        self.next_event_idx = []
        self.all_obs = []

        self.debug_frames = []


def randomize_object_poses(randomization_setting: dict, initial_conditions: list) -> list:
    result = dict(initial_conditions[0])

    for obj, ranges in randomization_setting.items():
        x = np.random.uniform(*ranges["x"])
        y = np.random.uniform(*ranges["y"])
        z = np.random.uniform(*ranges["z"])

        rx = np.random.uniform(*ranges.get("ori_x", (0, 0)))
        ry = np.random.uniform(*ranges.get("ori_y", (0, 0)))
        rz = np.random.uniform(*ranges.get("ori_z", (0, 0)))

        ori_pose = initial_conditions[0][obj]  # [x, y, z, qx, qy, qz, qw] xyzw
        orig_q = Rotation.from_quat(ori_pose[3:7])  # already xyzw
        rel_q = Rotation.from_euler("xyz", [rx, ry, rz])
        final_q = orig_q * rel_q
        q = final_q.as_quat()  # xyzw
        result[obj] = [x, y, z, q[3], q[0], q[1], q[2]]  # wxyz for Isaac Sim

        print(f"[{obj}] pos=({x:.3f}, {y:.3f}, {z:.3f})  quat(wxyz)=({q[3]:.3f}, {q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f})")
    return [result]

def is_success(info) -> bool:
    try:
        val = info["rubric"]["success"]
        v = val[0] if hasattr(val, "__len__") else val
        if isinstance(v, torch.Tensor):
            return bool(v.item())
        return bool(v)
    except (KeyError, TypeError, IndexError) as e:
        print(f"  [warn] Could not read info['rubric']['success']: {e}")
        return False


class MotionPlanner:
    def __init__(
        self,
        env,
        waypoints: list,
        recorder: ObsRecorder,
        robot_config: str = "franka_robotiq_2f_85.yml",
        steps_per_waypoint: int = 1,
        device: str = "cuda",
        
    ):
        self.env                = env
        self.waypoints          = waypoints
        self.steps_per_waypoint = steps_per_waypoint
        self.device             = device
        self.motion_gen         = self.setup_curobo(robot_config)
        self.recorder           = recorder

        self.GRIPPER_JOINT_IDX = 7
        self.GRIPPER_OPEN_VAL = 0
        self.GRIPPER_CLOSE_VAL = 1

        self.FINGER_CLOSED_Y = 0
        self.FINGER_OPEN_Y = 0.05
        
    def reset(self, object_poses):
        obs, info = self.env.reset(object_positions=object_poses)
        self.recorder.reset()
        self.add_obs(obs)
        return obs

    def setup_curobo(self, robot_cfg):

        import omni.usd
        from curobo.util.usd_helper import UsdHelper
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        stage = omni.usd.get_context().get_stage()
        assert stage is not None

        robot_prim_path = "/World/envs/env_0/robot"
        usd_help = UsdHelper()
        usd_help.stage = stage

        obstacle_world = usd_help.get_obstacles_from_stage(
            only_paths=["/World/envs/env_0"],
            reference_prim_path=robot_prim_path,
            ignore_substring=[
                robot_prim_path,                       
                "/World/defaultGroundPlane",
                "randomization",        
                "workspace_static",                   
                "OmniverseKitViewportCameraMesh",   
                "CameraModel",                        
                "env_light",                    
                "defaultLight",                 
                "Environment",                         
                "Render",
            ],
        )
        print("Obstacle meshes:", [m.name for m in (obstacle_world.mesh or [])])

        world_cfg = obstacle_world.get_collision_check_world()
        tensor_args = TensorDeviceType()

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            tensor_args,
            interpolation_dt=0.02,
        )
        motion_gen = MotionGen(motion_gen_config)
        motion_gen.warmup()

        return motion_gen
 
    def get_current_joints(self, obs) -> torch.Tensor:
        """Return (13,) joint positions on CUDA from policy obs dict."""

        return obs["joint_pos"].squeeze(0)[ISAAC_TO_CUROBO_IDX]  # (13,)
 
    def plan_trajectory(self, current_joints, target_pos, target_quat):
        """
        Plan a collision-free trajectory to target pose expressed in robot-base frame.
        cuRobo uses 8 DOF (arm7 + finger_joint); we pad finger_joint=0 for start state
        and return only the arm joints (T, 7).
        """

        from curobo.types.math import Pose
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
        from curobo.types.state import JointState

        goal_pose = Pose(position=target_pos, quaternion=target_quat)

        # cuRobo expects 8 DOF: append finger_joint=0
        joints_state = current_joints

        start_state = JointState.from_position(
            joints_state.unsqueeze(0),
            joint_names=self.motion_gen.joint_names,
        )

        result = self.motion_gen.plan_single(
            start_state=start_state,
            goal_pose=goal_pose,
            plan_config=MotionGenPlanConfig(max_attempts=5, time_dilation_factor=0.8),
        )
        if result.success[0]:
            return result.get_interpolated_plan().position[:, :self.GRIPPER_JOINT_IDX]  # (T, 7) arm joints only
        else:
            print(f"[cuRobo] Planning failed! status={result.status}")
            if result.position_error is not None:
                print(f"  position_error={result.position_error}")
            
        print("Use IK")
        ik_result = self.motion_gen.ik_solver.solve_single(goal_pose)
        print("IK success:", ik_result.success)
        print("IK solution:", ik_result.solution)
        print("IK error:", ik_result.error)

        if ik_result.success.item():
            return ik_result.solution.squeeze(0)[:, :self.GRIPPER_JOINT_IDX]
        else:
            return None
 
    def execute_trajectory(self, obs, traj, gripper_val):
        """
        Execute trajectory by stepping through waypoints.

        Args:
            env: Isaac environment
            obs: Current observation
            traj: Joint trajectory from cuRobo (T, 7)
            gripper_val: Gripper command (0=open, 1=close)
        """
        print(f"  Executing {traj.shape[0]} waypoints ({self.steps_per_waypoint} steps each)...")
        gripper_action = torch.tensor([gripper_val], device=self.device)

        for i in range(traj.shape[0]):
            action = traj[i].clone()
            action = torch.cat([action, gripper_action])  # (8,) = (7 arm joints + 1 gripper)
    
            # Debug print on first and last waypoint
            if i == 0:
                print(f"  [First waypoint] cuRobo traj[0]: {traj[i]}")
                print(f"  [First waypoint] action: {action}")
            if i == traj.shape[0] - 1:
                print(f"  [Last waypoint] cuRobo traj[-1]: {traj[i]}")
                print(f"  [Last waypoint] action: {action}")
            # Hold this waypoint for multiple steps to allow PD controller to converge
            for step_count in range(self.steps_per_waypoint):
                obs, rew, term, trunc, info = self.env.step(action.unsqueeze(0), expensive=True)
                self.add_obs(obs, action)

                if term[0] or trunc[0]:
                    return obs, info, True
        return obs, info, False
    
    def add_obs(self, obs, action=None):
        obs["gripper_pcd"] = self.get_gripper_pcd(obs)
        obs["state"] = self.get_state(obs)

        if action is not None:
            obs["abs_action"] = self.get_action(action)
        else:
            obs["abs_action"] = obs["state"]

        self.recorder.add(obs)

    def get_state(self, obs):
        # eef pos (3-dim) + eef rot (6-dim) + gripper width (1-dim) 
        from curobo.types.state import JointState

        joints = self.get_current_joints(obs)
        js     = JointState.from_position(joints.unsqueeze(0), joint_names=self.motion_gen.joint_names)
        fk     = self.motion_gen.compute_kinematics(js)

        # EE position (3,)
        ee_pos  = fk.ee_pos_seq[0].detach().cpu().numpy()

        # EE rotation as 6D representation (first two columns of rotation matrix)
        ee_quat = fk.ee_quat_seq[0].detach().cpu().numpy()  # wxyz
        R       = Rotation.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]]).as_matrix()

        ee_rot6d = R[:, :2].T.flatten()  # (6,) first two columns flattened

        # Gripper width (1,)
        gripper_val = obs["policy"]["gripper_pos"].item()
        gripper_width = self.get_gripper_width(gripper_val)

        return np.concatenate([ee_pos, ee_rot6d, np.array([gripper_width], dtype=np.float32)])

    def get_action(self, action: torch.Tensor) -> np.ndarray:
        """
        Get commanded EE action from joint action using FK.
        Returns: (10,) array = commanded eef pos (3) + eef rot 6d (6) + gripper width (1)
        """
        from curobo.types.state import JointState

        # Use arm joints only (7), pad gripper to full DOF
        arm_joints  = action[:self.GRIPPER_JOINT_IDX]
        full_joints = torch.cat([arm_joints, torch.zeros(6, device=self.device)])

        js = JointState.from_position(full_joints.unsqueeze(0), joint_names=self.motion_gen.joint_names)
        fk = self.motion_gen.compute_kinematics(js)

        # EE position (3,)
        ee_pos = fk.ee_pos_seq[0].detach().cpu().numpy()

        # EE rotation as 6D (first two columns of rotation matrix)
        ee_quat  = fk.ee_quat_seq[0].detach().cpu().numpy()  # wxyz
        R        = Rotation.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]]).as_matrix()
        ee_rot6d = R[:, :2].T.flatten()  # (6,)

        # Gripper width (1,)
        gripper_width = self.get_gripper_width(action[self.GRIPPER_JOINT_IDX].item())

        return np.concatenate([ee_pos, ee_rot6d, np.array([gripper_width], dtype=np.float32)])  # (10,)

    def execute_waypoints(self, obs, object_poses: dict):
        """
        Execute a sequence of waypoints defined relative to object poses.
        """
        info = {}
        
        for i, wp in enumerate(self.waypoints):
            obj = wp["object"]
            offset = wp["offset"]
            rel_quat_xyzw = wp["rel_quat"]  # x y z w
            grasp = wp["grasp"]
            subgoal = wp["subgoal"]

            obj_pose = object_poses[obj]  # [x, y, z, qw, qx, qy, qz]
            obj_pos = obj_pose[:3]
            obj_quat_wxyz = obj_pose[3:]  # [qw, qx, qy, qz]

            target_pos, target_quat = self.compute_waypoint_pose(
                obj_pos, obj_quat_wxyz, offset, rel_quat_xyzw,
            )

            print(f"\n[Waypoint {i}] obj={obj} grasp={grasp}")
            print(f"  target pos:  {target_pos}")
            print(f"  target quat: {target_quat}")
            # target_quat_test = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.device)  # wxyz pointing down
            traj = self.plan_trajectory(self.get_current_joints(obs), target_pos, target_quat)

            if traj is not None:
                obs, info, done = self.execute_trajectory(
                    obs, traj, grasp
                )
                
                if subgoal:
                    self.recorder.save_subgoal()

                if done:
                    print(f"  Early termination at waypoint {i}.")
                    return obs, info, True
            else:
                print(f"  Planning failed at waypoint {i}.")
                return obs, info, False

        if is_success(info):
            self.recorder.save_episode()

        return obs, info, False
    

    def get_gripper_pcd(self, obs) -> np.ndarray:
        """
        Get a 4-point point cloud representing the gripper state directly from cuRobo FK.
        Points:
        - grasp frame (EE)
        - left finger tip
        - right finger tip
        - top point above EE
        Returns: (4, 3) numpy array in world frame
        """
        from curobo.types.state import JointState

        joints = self.get_current_joints(obs)
        js = JointState.from_position(joints.unsqueeze(0), joint_names=self.motion_gen.joint_names)
        fk = self.motion_gen.compute_kinematics(js)

        ee_pos  = fk.ee_pos_seq[0].detach().cpu().numpy()   # (3,)
        ee_quat = fk.ee_quat_seq[0].detach().cpu().numpy()  # (4,) wxyz
        R = Rotation.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]]).as_matrix()

        # Gripper joint value, normalized to [0, 1] (0 = closed, 1 = open)
        gripper_val  = joints[self.GRIPPER_JOINT_IDX].item()
        gripper_width = self.get_gripper_width(gripper_val)

        offsets = np.array([
            [0.0,      0.0,  0.0 ],
            [0.0,  -gripper_width, 0.00],
            [0.0, gripper_width, 0.00],
            [0.0,      0.0, -0.05],
        ])  # (4, 3)

        points = (R @ offsets.T).T + ee_pos

        return points.astype(np.float32)


    def get_gripper_width(self, gripper_val):
        gripper_open  = self.GRIPPER_OPEN_VAL
        gripper_close = self.GRIPPER_CLOSE_VAL
        t = (gripper_val - gripper_close) / (gripper_open - gripper_close)
        t = np.clip(t, 0.0, 1.0)

        return self.FINGER_CLOSED_Y * (1-t) + self.FINGER_OPEN_Y * t
 
    def compute_waypoint_pose(self, obj_pos, obj_quat_wxyz, offset, rel_quat_wxyz):
        """
        Compute absolute world-frame pose for a waypoint.
        obj_pos: [x, y, z]
        obj_quat_wxyz: [qw, qx, qy, qz] - object orientation in world
        offset: [dx, dy, dz] - offset in world frame
        rel_quat_wxyz: [w, x, y, z] - relative rotation to apply on top of object orientation
        """

        # object orientation
        obj_R = Rotation.from_quat([obj_quat_wxyz[1], obj_quat_wxyz[2], obj_quat_wxyz[3], obj_quat_wxyz[0]])

        # absolute position = object pos + offset
        pos = torch.tensor([
            obj_pos[0] + offset[0],
            obj_pos[1] + offset[1],
            obj_pos[2] + offset[2],
        ], device=self.device, dtype=torch.float32).unsqueeze(0)  # (1, 3)

        # absolute orientation = obj_orientation * rel_quat
        rel_R = Rotation.from_quat([rel_quat_wxyz[1], rel_quat_wxyz[2], rel_quat_wxyz[3], rel_quat_wxyz[0]])  # xyzw
        abs_R = obj_R * rel_R
        q = abs_R.as_quat()  # xyzw
        quat = torch.tensor([q[3], q[0], q[1], q[2]], device=self.device, dtype=torch.float32).unsqueeze(0)  # (1, 4) wxyz

        return pos, quat

if __name__ == "__main__":
    args: DataArgs = tyro.cli(DataArgs)
    main(args)



