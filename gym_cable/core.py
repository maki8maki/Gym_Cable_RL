from abc import abstractmethod
from typing import Optional

import gymnasium as gym
from gymnasium import error


class GoalEnv(gym.Env):
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error("GoalEnv requires an observation space of type gym.spaces.Dict")
        for key in ["observation", "rgb_image", "depth_image"]:
            if key not in self.observation_space.spaces:
                raise error.Error(
                    'GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(key)
                )

    @abstractmethod
    def compute_reward(self, obs, goal, info):
        raise NotImplementedError

    @abstractmethod
    def compute_terminated(self, obs, goal, info):
        raise NotImplementedError

    @abstractmethod
    def compute_truncated(self, obs, goal, info):
        raise NotImplementedError
