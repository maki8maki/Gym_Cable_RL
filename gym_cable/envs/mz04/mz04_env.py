import numpy as np
from typing import Optional

from gym_cable.envs.robot_env import MujocoRobotEnv
from gym_cable.utils import rotations

DEFAULT_CAMERA_CONFIG = {
    "distance": 1.0,
    "azimuth": 132.0,
    "elevation": -20.0,
    "lookat": np.array([0.35, 0.0, 0.3]),
}


def get_base_mz04_env(RobotEnvClass: MujocoRobotEnv):
    """Factory function that returns a BaseMZ04Env class that inherits
    from MujocoRobotEnv.
    """

    class BaseMZ04Env(RobotEnvClass):
        """Superclass for all MZ04 environments."""

        def __init__(
            self,
            target_offset,
            obj_position_range,
            obj_posture_range,
            position_random,
            posture_random,
            distance_threshold,
            rotation_threshold,
            rot_weight,
            **kwargs
        ):
            """Initializes a new MZ04 environment.

            Args:
                model_path (string): path to the environments XML file
                n_substeps (int): number of substeps the simulation runs on every call to step
                target_offset (float or array with 3 elements): offset of the target
                obj_position_range (float): range of a uniform distribution for sampling initial object positions
                obj_postures_range (int): range of a uniform distribution for sampling initial object postures
                distance_threshold (float): the threshold after which a goal is considered achieved
                initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
            """

            self.target_offset = target_offset
            self.obj_position_range = obj_position_range
            self.obj_posture_range = obj_posture_range
            self.position_random = position_random
            self.posture_random = posture_random
            self.distance_threshold = distance_threshold
            self.rotation_threshold = rotation_threshold
            self.rot_weight = rot_weight

            super().__init__(n_actions=6, **kwargs)

        # GoalEnv methods
        # ----------------------------

        def compute_reward(self, obs, goal, info):
            if self.terminated:
                reward = 10
            elif self.truncated:
                reward = -1
            else:
                position_err, posture_err = self._utils.calc_err_norm(obs, goal)
                err_norm = position_err + posture_err * self.rot_weight
                reward = -err_norm
            return reward
        
        def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
        ):
            obs, info = super().reset(seed=seed)
            if options is not None:
                try:
                    r = 1 - options["diff_ratio"]
                    assert(0 <= r and r <= 1)
                    pos_c, quat_c = obs['observation'][:3], rotations.euler2quat(obs['observation'][3:])
                    pos_g, quat_g = self.goal[:3], rotations.euler2quat(self.goal[3:])
                    pos_t = (1 - r) * pos_c + r * pos_g
                    quat_t = rotations.quat_slerp(quat_c, quat_g, r)
                    _ = self._utils.set_site_to_xpos(self.model, self.data, self.site_name, self.joint_names, pos_t, quat_t)
                except Exception as e:
                    print(e)
            obs = self._get_obs()
            if self.render_mode == "human":
                self.render()

            return obs, info

        # RobotEnv methods
        # ----------------------------

        def _set_action(self, action):
            assert action.shape == (6,)
            action = action.copy() # ensure that we don't change the action outside of this scope
            pos_ctrl, rot_ctrl = action[:3], action[3:]
            
            pos_ctrl *= 0.05 # limit maximum change in position
            rot_ctrl *= np.deg2rad(10) # limit maximum change in rotation

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
            cable_end_pos = self.data.body('B_last').xpos
            cable_end_mat = self.data.body('B_last').xmat.reshape(3, 3)
            goal_pos = cable_end_pos + cable_end_mat.T @ self.target_offset
            goal_rot = rotations.quat2euler(self.data.body('B_last').xquat)
            return np.concatenate([goal_pos, goal_rot])

        def _is_success(self, obs, goal):
            position_err, posture_err = self._utils.calc_err_norm(obs, goal)
            return (position_err < self.distance_threshold) and (posture_err < self.rotation_threshold)

    return BaseMZ04Env

class MujocoMZ04Env(get_base_mz04_env(MujocoRobotEnv)):
    def __init__(self, site_name, joint_names, default_camera_config: dict = DEFAULT_CAMERA_CONFIG, **kwargs):
        super().__init__(default_camera_config=default_camera_config, **kwargs)
        
        self.site_name = site_name
        self.joint_names = joint_names

    def _step_callback(self):
        self._mujoco.mj_forward(self.model, self.data)

    def _set_action(self, action):
        action = super()._set_action(action)

        # Apply action to simulation.
        self.ik_success = self._utils.ik_set_action(self.model, self.data, action, self.site_name, self.joint_names)

    def generate_mujoco_observations(self):
        # positions and rotation(eular)
        ee_pos = self._utils.get_site_xpos(self.model, self.data, "robot:end_effector")
        ee_rot = rotations.mat2euler(self._utils.get_site_xmat(self.model, self.data, "robot:end_effector"))

        # RGB and Depth images
        rgb_image =  self.mujoco_renderer.render(render_mode="rgb_array", camera_name="robot:camera")
        depth_image = self.mujoco_renderer.render(render_mode="depth_array", camera_name="robot:camera")

        # Get the distances to the near and far clipping planes.
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent

        # Convert from [0 1] to depth in units of length, see links below:
        # http://stackoverflow.com/a/6657284/1461210
        # https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
        depth_image = near / (1 - depth_image * (1 - near / far))
        
        # Set pixel values outside the range to 0
        depth_image[(depth_image < self.depth_range[0]) | (depth_image > self.depth_range[1])] = 0

        return (ee_pos, ee_rot, rgb_image, np.expand_dims(depth_image, 2))

    def _render_callback(self):
        self._mujoco.mj_forward(self.model, self.data)
        
    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Randomize start position and posture of object.
        if self.position_random:
            diff = self.np_random.uniform(-self.obj_position_range, self.obj_position_range)
            circuit_pos = np.copy(self.initial_circuit_pos)
            circuit_pos[1] += diff
            self._utils.set_joint_qpos(self.model, self.data, "circuit:joint", circuit_pos)
            self.model.body("B_first").pos = np.copy(self.initial_cable_pos)
            self.model.body("B_first").pos[1] += diff
        if self.posture_random:
            diff = self.np_random.uniform(0, self.obj_posture_range)
            self.model.body('B_10').quat = rotations.euler2quat(np.deg2rad(np.array([0, diff-90, 0])))

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._mujoco.mj_forward(self.model, self.data)
    
    def set_threshold(self, distance_threshold=None, rotation_threshold=None):
        if distance_threshold is not None:
            self.distance_threshold = distance_threshold
        if rotation_threshold is not None:
            self.rotation_threshold = rotation_threshold
