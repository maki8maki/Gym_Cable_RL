import os

import hydra
import numpy as np
import torch as th
from omegaconf import OmegaConf
from tqdm import tqdm

from config import SB3Config
from gym_cable.utils.mujoco_utils import calc_err_norm
from utils import set_seed


@hydra.main(config_path="conf/", config_name="sb3", version_base=None)
def main(_cfg: OmegaConf):
    cfg = SB3Config.convert(_cfg)
    set_seed(0)

    model = cfg.model.load(os.path.join("model", f"{cfg.basename}.zip"))
    policy = model.policy
    policy.set_training_mode(False)

    env = cfg.env

    ac_norms = []
    pos_err = []
    ori_err = []

    key = "observation"
    for _ in tqdm(range(100)):
        obs, _ = env.reset()
        for _ in range(cfg.nsteps):
            obs = th.tensor(obs)
            action, _ = policy.predict(obs, deterministic=True)
            next_obs, _, terminated, truncated, _ = env.step(action)
            ac_norms.append(np.linalg.norm(action) / np.sqrt(6))

            obs = obs.numpy()[20:]
            obs_mean = 0.5 * (env.old_observation_space[key].high + env.old_observation_space[key].low)
            obs_halfwidth = 0.5 * (env.old_observation_space[key].high - env.old_observation_space[key].low)
            obs = obs * obs_halfwidth + obs_mean
            position_err, posture_err = calc_err_norm(obs, env.unwrapped.goal)
            pos_err.append(position_err)
            ori_err.append(posture_err)

            if terminated or truncated:
                break
            else:
                obs = next_obs

    np.savez("data/execute", ac=ac_norms, pos=pos_err, ori=ori_err)
    env.close()


if __name__ == "__main__":
    main()
