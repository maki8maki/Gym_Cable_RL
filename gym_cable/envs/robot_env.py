import os
from typing import Optional

import numpy as np
from gymnasium import error, spaces
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

from gym_cable.core import GoalEnv

try:
    import mujoco

    from gym_cable.utils import mujoco_utils
except ImportError as e:
    MUJOCO_IMPORT_ERROR = e
else:
    MUJOCO_IMPORT_ERROR = None

DEFAULT_SIZE = 480


class BaseRobotEnv(GoalEnv):
    """Superclass for all MuJoCo fetch and hand robotic environments."""

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 25,
    }

    def __init__(
        self,
        model_path: str,
        initial_qpos,
        n_actions: int,
        n_substeps: int,
        render_mode: Optional[str] = None,
        depth_min: float = 0.07,
        depth_max: float = 0.5,
        width: int = DEFAULT_SIZE,
        height: int = DEFAULT_SIZE,
    ):
        if model_path.startswith("/"):
            self.fullpath = model_path
        else:
            self.fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(self.fullpath):
            raise OSError(f"File {self.fullpath} does not exist")

        self.n_substeps = n_substeps

        self.initial_qpos = initial_qpos

        self.depth_range = np.array([depth_min, depth_max])

        self.width = width
        self.height = height
        self._initialize_simulation()

        self.goal = np.zeros(0)
        obs = self._get_obs()

        assert (
            int(np.round(1.0 / self.dt)) == self.metadata["render_fps"]
        ), f'Expected value: {int(np.round(1.0 / self.dt))}, Actual value: {self.metadata["render_fps"]}'

        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype="float64"),
                rgb_image=spaces.Box(0, 255, shape=obs["rgb_image"].shape, dtype="uint8"),
                depth_image=spaces.Box(0, 255, shape=obs["depth_image"].shape, dtype="uint8"),
            )
        )

        self.render_mode = render_mode

    # Env methods
    # ----------------------------
    def compute_terminated(self, obs, goal, info):
        return info["is_success"] and not (self.truncated)

    def compute_truncated(self, obs, goal, info):
        info["ik_success"] = self.ik_success
        info["contacted"] = self.data.ncon > 10
        position_err, posture_err = self._utils.calc_err_norm(obs, goal)
        is_far = (position_err > 0.2) or (posture_err > np.deg2rad(45))
        info["is_far"] = is_far
        return (not info["ik_success"]) or info["contacted"] or info["is_far"]

    def step(self, action):
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        self._mujoco_step(action)

        self._step_callback()

        if self.render_mode == "human":
            self.render()
        obs = self._get_obs()

        info = {
            "is_success": self._is_success(obs["observation"], self.goal),
        }

        self.truncated = self.compute_truncated(obs["observation"], self.goal, info)
        self.terminated = self.compute_terminated(obs["observation"], self.goal, info)

        reward = self.compute_reward(obs["observation"], self.goal, info)

        return obs, reward, self.terminated, self.truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        return obs, {}

    # Extension methods
    # ----------------------------
    def _mujoco_step(self, action):
        """Advance the mujoco simulation.

        Override depending on the python binginds, either mujoco or mujoco_py
        """
        raise NotImplementedError

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.

        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        return True

    def _initialize_simulation(self):
        """Initialize MuJoCo simulation data structures mjModel and mjData."""
        raise NotImplementedError

    def _get_obs(self):
        """Returns the observation."""
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation."""
        raise NotImplementedError()

    def _is_success(self, obs, goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it."""
        raise NotImplementedError()

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment.

        Can be used to configure initial state and extract information from the simulation.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering.

        Can be used to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation.

        Can be used to enforce additional constraints on the simulation state.
        """
        pass


class MujocoRobotEnv(BaseRobotEnv):
    """Robot base class for fetch and hand environment versions that depend on new mujoco bindings from Deepmind."""

    def __init__(self, default_camera_config: Optional[dict] = None, **kwargs):
        if MUJOCO_IMPORT_ERROR is not None:
            raise error.DependencyNotInstalled(f"{MUJOCO_IMPORT_ERROR}. (HINT: you need to install mujoco)")

        self._mujoco = mujoco
        self._utils = mujoco_utils

        self.default_camera_config = default_camera_config

        super().__init__(**kwargs)

    def _initialize_simulation(self):
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        self._env_setup(initial_qpos=self.initial_qpos)
        self.initial_time = self.data.time
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)
        self.initial_circuit_pos = self._utils.get_joint_qpos(self.model, self.data, "circuit:joint")
        self.initial_cable_pos = np.copy(self.model.body("B_first").pos)

        self.mujoco_renderer = MujocoRenderer(self.model, self.data, self.default_camera_config)

    def _reset_sim(self):
        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        mujoco.mj_forward(self.model, self.data)
        return super()._reset_sim()

    def render(self):
        """Render a frame of the MuJoCo simulation.

        Returns:
            rgb image (np.ndarray): if render_mode is "rgb_array", return a 3D image array.
        """
        self._render_callback()
        return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        """Close contains the code necessary to "clean up" the environment.

        Terminates any existing WindowViewer instances in the Gymnaisum MujocoRenderer.
        """
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()

    @property
    def dt(self):
        """Return the timestep of each Gymanisum step."""
        return self.model.opt.timestep * self.n_substeps

    def _mujoco_step(self, action):
        # self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        return
