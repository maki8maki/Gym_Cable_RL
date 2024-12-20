from typing import Optional

import cv2  # noqa: F401
import numpy as np

from gym_cable.envs.robot_env import MujocoRobotEnv
from gym_cable.utils import rotations

DEFAULT_CAMERA_CONFIG = {
    "distance": 0.8,
    "azimuth": 135.0,
    "elevation": -30.0,
    "lookat": np.array([0.25, 0.0, 0.45]),
}


def get_base_mz04_env(RobotEnvClass: MujocoRobotEnv):
    """Factory function that returns a BaseMZ04Env class that inherits
    from MujocoRobotEnv.
    """

    class BaseMZ04Env(RobotEnvClass):
        """Superclass for all MZ04 environments."""

        def __init__(
            self,
            target_offset: np.ndarray,
            obj_position_range: float,
            obj_posture_range: float,
            position_random: bool,
            posture_random: bool,
            with_continuous: bool,
            distance_threshold: float,
            rotation_threshold: float,
            rot_weight: float,
            cable_width: float = None,
            circuit_width: float = None,
            **kwargs,
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
            self.with_continuous = with_continuous
            self.distance_threshold = distance_threshold
            self.rotation_threshold = rotation_threshold
            self.rot_weight = rot_weight

            n_actions = 6

            super().__init__(n_actions=n_actions, **kwargs)

            # *** ケーブル等の寸法のみを変更 ***
            if cable_width is not None and circuit_width is not None:
                w = cable_width / 2

                # ケーブル
                id = 0
                while True:
                    try:
                        self.model.geom("G" + str(id)).size[1] = w
                        id += 1
                    except KeyError:
                        break
                boneadr = self.model.skin("Skin").boneadr[0]
                bonenum = self.model.skin("Skin").bonenum[0]
                for i in range(bonenum):
                    self.model.skin_bonebindpos[boneadr + i][1] = w * ((-1) ** (i % 2 + 1))

                # コネクタ
                self.model.geom("connector").size[1] = w + 0.0017

                # 基板
                self.model.geom("board").size[1] = circuit_width / 2

        # GoalEnv methods
        # ----------------------------

        def compute_reward(self, obs: np.ndarray, goal: np.ndarray, info: dict):
            if self.terminated and not self.with_continuous:
                reward = 10
            elif self.truncated:
                reward = -1
            else:
                position_err, posture_err = self._utils.calc_err_norm(obs, goal)
                err_norm = position_err + posture_err * self.rot_weight
                reward = -err_norm
            return reward

        def compute_terminated(self, obs: np.ndarray, goal: np.ndarray, info: dict):
            if self.with_continuous:
                terminated = False
            else:
                terminated = super().compute_terminated(obs, goal, info)
            return terminated

        def compute_truncated(self, obs: np.ndarray, goal: np.ndarray, info: dict):
            truncated = super().compute_truncated(obs, goal, info)
            if truncated:
                info.update([("is_success", False)])
            return truncated

        def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
        ):
            obs, info = super().reset(seed=seed)

            obs = self._get_obs()
            if self.render_mode == "human":
                self.render()

            return obs, info

        # RobotEnv methods
        # ----------------------------

        def _set_action(self, action: np.ndarray):
            assert action.shape == (6,)
            action = action.copy()  # ensure that we don't change the action outside of this scope
            pos_ctrl, rot_ctrl = action[:3], action[3:]

            pos_ctrl *= 0.05  # limit maximum change in position
            rot_ctrl *= np.deg2rad(10)  # limit maximum change in rotation

            quat_ctrl = rotations.euler2quat(rot_ctrl)
            action = np.concatenate([pos_ctrl, quat_ctrl])

            return action

        def _get_obs(self):
            ee_pos, ee_rot, rgb_image, depth_image = self.generate_mujoco_observations()

            obs = np.concatenate([ee_pos, ee_rot])

            return {"observation": obs.copy(), "rgb_image": rgb_image.copy(), "depth_image": depth_image.copy()}

        def generate_mujoco_observations(self):

            raise NotImplementedError

        def _get_gripper_xpos(self):

            raise NotImplementedError

        def _sample_goal(self):
            cable_end_pos = self._utils.get_site_xpos(self.model, self.data, "S_last")
            cable_end_mat = self._utils.get_site_xmat(self.model, self.data, "S_last")
            diff = np.dot(cable_end_mat, np.reshape(self.target_offset, (3, 1)))
            goal_pos = cable_end_pos + np.reshape(diff, (3,))
            goal_rot = rotations.mat2euler(cable_end_mat)
            return np.concatenate([goal_pos, goal_rot])

        def _is_success(self, obs: np.ndarray, goal: np.ndarray):
            position_err, posture_err = self._utils.calc_err_norm(obs, goal)
            return (position_err < self.distance_threshold) and (posture_err < self.rotation_threshold)

    return BaseMZ04Env


class MujocoMZ04Env(get_base_mz04_env(MujocoRobotEnv)):
    def __init__(
        self,
        site_name: str,
        joint_names: list[str],
        default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
        **kwargs,
    ):
        super().__init__(default_camera_config=default_camera_config, **kwargs)

        self.site_name = site_name
        self.joint_names = joint_names

    def _step_callback(self):
        self._mujoco.mj_forward(self.model, self.data)

    def _set_action(self, action: np.ndarray):
        action = super()._set_action(action)

        # Apply action to simulation.
        self.ik_success = self._utils.ik_set_action(self.model, self.data, action, self.site_name, self.joint_names)

    def generate_mujoco_observations(self):
        # positions and rotation(euler)
        ee_pos = self._utils.get_site_xpos(self.model, self.data, "robot:end_effector")
        ee_rot = rotations.mat2euler(self._utils.get_site_xmat(self.model, self.data, "robot:end_effector"))

        # RGB and Depth images
        rgb_image = self.mujoco_renderer.render(render_mode="rgb_array", camera_name="robot:camera")
        depth_image = self.mujoco_renderer.render(render_mode="depth_array", camera_name="robot:camera")

        # Get the distances to the near and far clipping planes.
        extent = self.model.stat.extent
        near = self.model.vis.map.znear * extent
        far = self.model.vis.map.zfar * extent

        # Convert from [0 1] to depth in units of length, see links below:
        # http://stackoverflow.com/a/6657284/1461210
        # https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
        depth_image = near / (1 - depth_image * (1 - near / far))

        # 試し
        # rgb_image_right = self.mujoco_renderer.render(render_mode="rgb_array", camera_name="robot:camera_right")
        # gray_left = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        # gray_right = cv2.cvtColor(rgb_image_right, cv2.COLOR_RGB2GRAY)
        # stereo = cv2.StereoBM.create(numDisparities=128, blockSize=5)
        # disparity = stereo.compute(gray_left, gray_right)
        # depth_image = np.zeros(shape=gray_left.shape).astype(float)
        # baseline = 18.0 / 1000
        # ratio = disparity.shape[1] / 2.0 / np.tan(np.deg2rad(58) / 2)
        # depth_image[disparity > 0] = (baseline * ratio * 16) / disparity[disparity > 0]

        # Set pixel values outside the range to 0 and Scale [0 255]
        depth_image[(depth_image < self.depth_range[0]) | (depth_image > self.depth_range[1])] = 0
        depth_image = depth_image / self.depth_range[1] * 255

        return (ee_pos, ee_rot, rgb_image, np.expand_dims(depth_image.astype(np.uint8), 2))

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
            self.model.body("fixing").pos = circuit_pos
            self.model.body("B_first").pos = np.copy(self.initial_cable_pos)
            self.model.body("B_first").pos[1] += diff
        if self.posture_random:
            diff = self.np_random.uniform(0, self.obj_posture_range)
            self.model.body("B_11").quat = rotations.euler2quat(np.deg2rad(np.array([180, 90 - diff, 0])))

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _env_setup(self, initial_qpos: dict[str, float]):
        for name, value in initial_qpos.items():
            self._utils.set_joint_qpos(self.model, self.data, name, value)
        self._mujoco.mj_forward(self.model, self.data)
