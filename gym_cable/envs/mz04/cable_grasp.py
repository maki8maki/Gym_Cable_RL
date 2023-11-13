import os

from gymnasium.utils.ezpickle import EzPickle
from gym_cable.envs.mz04.mz04_env import MujocoMZ04Env

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("mz04", "grasp.xml")


class MujocoMZ04CableGraspEnv(MujocoMZ04Env, EzPickle):
    def __init__(self, **kwargs):
        initial_qpos = {
            "robot:j1_joint": 0,
        }
        MujocoMZ04Env.__init__(
            self,
            model_path=MODEL_XML_PATH,
            n_substeps=20,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            width=1280,
            height=720,
            **kwargs,
        )
        EzPickle.__init__(self, **kwargs)
