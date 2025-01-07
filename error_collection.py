import os

import hydra
import numpy as np
import torch as th
from omegaconf import OmegaConf

from config import SB3Config
from gym_cable.utils import rotations as rot
from utils import set_seed


@hydra.main(config_path="conf/", config_name="sb3", version_base=None)
def main(_cfg: OmegaConf):
    cfg = SB3Config.convert(_cfg)
    set_seed(0)

    model = cfg.model.load(os.path.join("model", f"{cfg.basename}.zip"))
    policy = model.policy
    policy.set_training_mode(False)

    env = cfg.env

    err = []
    for _ in range(100):
        obs, _ = env.reset()
        success_num = 0
        for _ in range(cfg.nsteps):
            obs = th.tensor(obs)
            action, _ = policy.predict(obs, deterministic=True)
            next_obs, _, terminated, truncated, info = env.step(action)
            if info["is_success"]:
                success_num += 1
            if terminated or truncated:
                obs = obs.numpy()[20:]
                key = "observation"
                obs_mean = 0.5 * (env.old_observation_space[key].high + env.old_observation_space[key].low)
                obs_halfwidth = 0.5 * (env.old_observation_space[key].high - env.old_observation_space[key].low)
                obs = obs * obs_halfwidth + obs_mean
                diff = np.array(env.unwrapped.goal[:3] - obs[:3])
                mat = rot.euler2mat(env.unwrapped.goal[3:])
                inv_mat = np.linalg.inv(mat)
                err.append((inv_mat @ diff)[2])
                break
            else:
                obs = next_obs

    np.save(os.path.join("data", "error"), err)

    env.close()


if __name__ == "__main__":
    main()
