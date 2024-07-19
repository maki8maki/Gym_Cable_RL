from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch as th
from torch import nn

from agents.utils import FE


class FEWrapper(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, model: FE, trans: nn.Module):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)

        self.model = model
        self.trans = trans
        self.device = model.device

        self.old_observation_space = deepcopy(self.env.observation_space)
        self.observation_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(self.model.hidden_dim + self.old_observation_space["observation"].shape[0],),
            dtype=np.float32,
        )

        self.model.eval()

    def normalize_observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # 連続値の状態を[-1,1]の範囲に正規化
        normalized_obs = {}
        for key in obs.keys():
            obs_mean = 0.5 * (self.old_observation_space[key].high + self.old_observation_space[key].low)
            obs_halfwidth = 0.5 * (self.old_observation_space[key].high - self.old_observation_space[key].low)
            normalized_obs[key] = (obs[key].astype(np.float32) - obs_mean) / obs_halfwidth
        return normalized_obs

    def convert_observation(
        self, obs: dict[str, np.ndarray], image_list: list[str] = ["rgb_image", "depth_image"]
    ) -> dict[str, np.ndarray]:
        normalized_obs = self.normalize_observation(obs)
        if len(image_list) > 0:
            image = normalized_obs[image_list[0]] * 0.5 + 0.5  # [-1 1] -> [0 1]
            for name in image_list[1:]:
                image = np.concatenate([image, normalized_obs[name] * 0.5 + 0.5], axis=2)
            image = th.tensor(self.trans(image), dtype=th.float, device=self.device)
            hs = self.model.forward(image.unsqueeze(0)).cpu().squeeze().detach().numpy()
            state = np.concatenate([hs, normalized_obs["observation"]])
            return state
        else:
            return normalized_obs["observation"]

    def observation(self, observation: gym.spaces.Dict) -> np.ndarray:
        return self.convert_observation(observation)
