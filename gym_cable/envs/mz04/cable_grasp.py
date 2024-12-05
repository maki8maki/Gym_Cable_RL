import os

import gymnasium as gym
import numpy as np
from gymnasium.utils.ezpickle import EzPickle

from gym_cable.envs.mz04.mz04_env import MujocoMZ04Env

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("mz04", "mjmodel.xml")


class MujocoMZ04CableGraspEnv(MujocoMZ04Env, EzPickle):
    def __init__(self, **kwargs):
        initial_qpos = {
            "robot:j1_joint": 0.0,
            "robot:j2_joint": 1.53,
            "robot:j3_joint": -0.401,
            "robot:j4_joint": 0.0,
            "robot:j5_joint": -1.17,
            "robot:j6_joint": 0.0,
        }
        MujocoMZ04Env.__init__(
            self,
            model_path=MODEL_XML_PATH,
            n_substeps=20,
            target_offset=np.array([-0.015, 0.0, 0.002]),
            obj_position_range=0.04,  # m
            obj_posture_range=20,  # deg
            cable_width_range=np.array([0.01, 0.03]),  # m
            distance_threshold=0.005,  # m
            rotation_threshold=np.deg2rad(3),  # rad
            rot_weight=0.25,
            initial_qpos=initial_qpos,
            width=848,
            height=480,
            site_name="robot:end_effector",
            joint_names=[
                "robot:j1_joint",
                "robot:j2_joint",
                "robot:j3_joint",
                "robot:j4_joint",
                "robot:j5_joint",
                "robot:j6_joint",
            ],
            **kwargs,
        )
        EzPickle.__init__(self, **kwargs)

        # 観測範囲を限定
        self.observation_space["observation"] = gym.spaces.Box(
            low=np.array([-0.0, -1.0, -0.0, -np.pi, -np.pi, -np.pi]),
            high=np.array([2.0, 1.0, 2.0, np.pi, np.pi, np.pi]),
            dtype="float64",
        )
