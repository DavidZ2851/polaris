import torch
from scipy.spatial.transform import Rotation

from data_utils import ObsRecorder
from eval_utils import is_success

class MotionPlanner:
    def __init__(
        self,
        env,
        waypoints: list,
        recorder: ObsRecorder,
        device: str = "cuda",
        
    ):
        self.env  = env
        self.waypoints = waypoints
        self.device = device
        self.recorder = recorder
        
    def reset(self, object_poses):
        obs, info = self.env.reset(object_positions=object_poses)
        self.recorder.reset()
        return obs
 
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
            joint_names=self.env.motion_gen.joint_names,
        )

        result = self.env.motion_gen.plan_single(
            start_state=start_state,
            goal_pose=goal_pose,
            plan_config=MotionGenPlanConfig(max_attempts=5, time_dilation_factor=0.8),
        )
        if result.success[0]:
            return result.get_interpolated_plan().position[:, :self.env.GRIPPER_JOINT_IDX]  # (T, 7) arm joints only
        else:
            print(f"[cuRobo] Planning failed! status={result.status}")
            if result.position_error is not None:
                print(f"  position_error={result.position_error}")
            
        print("Use IK")
        ik_result = self.env.motion_gen.ik_solver.solve_single(goal_pose)
        print("IK success:", ik_result.success)
        print("IK solution:", ik_result.solution)
        print("IK error:", ik_result.error)

        if ik_result.success.item():
            return ik_result.solution.squeeze(0)[:, :self.env.GRIPPER_JOINT_IDX]
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
        print(f"  Executing {traj.shape[0]} waypoints")
        gripper_action = torch.tensor([gripper_val], device=self.device)

        for i in range(traj.shape[0]):
            action = traj[i].clone()
            action = torch.cat([action, gripper_action])  # (8,) = (7 arm joints + 1 gripper)

            self.add_obs(obs)

            if i == 0:
                print(f"  [First waypoint] cuRobo traj[0]: {traj[i]}")
                print(f"  [First waypoint] action: {action}")
            if i == traj.shape[0] - 1:
                print(f"  [Last waypoint] cuRobo traj[-1]: {traj[i]}")
                print(f"  [Last waypoint] action: {action}")
           
            obs, rew, term, trunc, info = self.env.step(action.unsqueeze(0), expensive=True)
            
            if term[0] or trunc[0]:
                return obs, info, True
    
        return obs, info, False
    
    def add_obs(self, obs):
        self.recorder.add(obs)

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
            traj = self.plan_trajectory(self.env.get_current_joints(obs), target_pos, target_quat)

            if traj is not None:
                obs, info, done = self.execute_trajectory(
                    obs, traj, grasp
                )
                
                if subgoal:
                    self.recorder.save_subgoal()

                if done:
                    print(f"  Early termination at waypoint {i}.")
                    self.add_obs(obs) # record terminal state ONCE here
                    return obs, info, True
            else:
                print(f"  Planning failed at waypoint {i}.")
                return obs, info, False
        
        # record terminal state ONCE after all waypoints complete
        self.add_obs(obs)

        if is_success(info):
            self.recorder.save_episode()

        return obs, info, False
 
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