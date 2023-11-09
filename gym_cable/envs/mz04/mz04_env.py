import numpy as np

from gym_cable.envs.robot_env import MujocoRobotEnv
from gym_cable.utils import rotations

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 132.0,
    "elevation": -14.0,
    "lookat": np.array([1.3, 0.75, 0.55]),
}


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

def get_base_mz04_env(RobotEnvClass: MujocoRobotEnv):
    """Factory function that returns a BaseMZ04Env class that inherits
    from MujocoRobotEnv.
    """

    class BaseMZ04Env(RobotEnvClass):
        """Superclass for all MZ04 environments."""

        def __init__(
            self,
            target_offset,
            obj_range,
            target_range,
            distance_threshold,
            **kwargs
        ):
            """Initializes a new MZ04 environment.

            Args:
                model_path (string): path to the environments XML file
                n_substeps (int): number of substeps the simulation runs on every call to step
                target_offset (float or array with 3 elements): offset of the target
                obj_range (float): range of a uniform distribution for sampling initial object positions
                target_range (float): range of a uniform distribution for sampling a target
                distance_threshold (float): the threshold after which a goal is considered achieved
                initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            """

            self.target_offset = target_offset
            self.obj_range = obj_range
            self.target_range = target_range
            self.distance_threshold = distance_threshold

            super().__init__(n_actions=6, **kwargs)

        # GoalEnv methods
        # ----------------------------

        def compute_reward(self, achieved_goal, goal, info):
            # Compute distance between goal and the achieved goal.
            return goal_distance(achieved_goal, goal)

        # RobotEnv methods
        # ----------------------------

        def _set_action(self, action):
            assert action.shape == (6,)
            action = action.copy() # ensure that we don't change the action outside of this scope
            pos_ctrl, rot_ctrl = action[:3], action[3]
            
            pos_ctrl *= 0.05 # limit maximum change in position
            rot_ctrl *= np.pi * 10.0 / 180.0 # limit maximum change in rotation

            quat_ctrl = rotations.euler2quat(rot_ctrl)
            action = np.concatenate([pos_ctrl, quat_ctrl])

            return action

        def _get_obs(self):
            ee_pos, ee_rot, rgb_image, depth_image = self.generate_mujoco_observations()

            obs = np.concatenate([ee_pos, ee_rot])

            return {
                "observation": obs.copy(),
                "rgb_image": rgb_image.copy(),
                "depth_image": depth_image.copy()
            }

        def generate_mujoco_observations(self):

            raise NotImplementedError

        def _get_gripper_xpos(self):

            raise NotImplementedError

        def _sample_goal(self):
            goal = np.random.uniform(0, 1, (3,))
            # if self.has_object:
            #     goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            #         -self.target_range, self.target_range, size=3
            #     )
            #     goal += self.target_offset
            #     goal[2] = self.height_offset
            #     if self.target_in_the_air and self.np_random.uniform() < 0.5:
            #         goal[2] += self.np_random.uniform(0, 0.45)
            # else:
            #     goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            #         -self.target_range, self.target_range, size=3
            #     )
            return goal.copy()

        def _is_success(self, achieved_goal, desired_goal):
            d = goal_distance(achieved_goal, desired_goal)
            return (d < self.distance_threshold).astype(np.float32)

    return BaseMZ04Env

class MujocoMZ04Env(get_base_mz04_env(MujocoRobotEnv)):
    def __init__(self, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs):
        super().__init__(default_camera_config=default_camera_config, **kwargs)

    def _step_callback(self):
        self._mujoco.mj_forward(self.model, self.data)

    def _set_action(self, action):
        action = super()._set_action(action)

        # Apply action to simulation.
        self._utils.mocap_set_action(self.model, self.data, action)

    def generate_mujoco_observations(self):
        # positions and rotation(eular)
        ee_pos = self._utils.get_site_xpos(self.model, self.data, "robot:end_effector")
        ee_rot = rotations.mat2euler(self._utils.get_site_xmat(self.model, self.data, "robot:end_effector"))

        # RGB and Depth images
        rgb_image =  self.mujoco_renderer.render(render_mode="rgb_array", camera_name="robot:camera")
        depth_image = self.mujoco_renderer.render(render_mode="depth_array", camera_name="robot:camera")
        
        # Set pixel values outside the range to 0
        depth_image[(depth_image < self.depth_range[0]) | (depth_image > self.depth_range[1])] = 0

        return (ee_pos, ee_rot, rgb_image, depth_image)

    def _render_callback(self):
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # # Randomize start position of object.
        # if self.has_object:
        #     object_xpos = self.initial_gripper_xpos[:2]
        #     while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
        #         object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
        #     object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object0:joint")
        #     assert object_qpos.shape == (7,)
        #     object_qpos[:2] = object_xpos
        #     self._utils.set_joint_qpos(self.model, self.data, "object0:joint", object_qpos)

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._utils.reset_mocap_welds(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)

        # # Move end effector into position.
        # gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        # gripper_rotation = np.array([1.0, 0.0, 1.0, 0.0])
        # self._utils.set_mocap_pos(self.model, self.data, "robot0:mocap", gripper_target)
        # self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap", gripper_rotation)
        # for _ in range(10):
        #     self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        # # Extract information for sampling goals.
        # self.initial_gripper_xpos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        # if self.has_object:
        #     self.height_offset = self._utils.get_site_xpos(self.model, self.data, "object0")[2]