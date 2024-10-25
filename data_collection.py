import os

import hydra
import numpy as np
import torch as th
from omegaconf import OmegaConf
from tqdm import tqdm

from config import SB3Config
from utils import set_seed


@hydra.main(config_path="conf/", config_name="sb3", version_base=None)
def main(_cfg: OmegaConf):
    cfg = SB3Config.convert(_cfg)
    set_seed(0)

    model = cfg.model.load(os.path.join("logs", "SB3_SAC_trainedVAE", "20241018-1319", f"{cfg.basename}.zip"))
    policy = model.policy
    policy.set_training_mode(False)

    env = cfg.env

    images = []
    poses = []
    size = 1000
    obs, _ = env.reset()
    step = 0
    with tqdm(total=size) as pbar:
        while len(images) < size:
            obs = th.tensor(obs)
            # action, _ = policy.predict(obs, deterministic=False)
            action = env.action_space.sample()
            next_obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated or step >= cfg.nsteps:
                obs, _ = env.reset()
                step = 0
            else:
                step += 1
                obs = next_obs
                img = env.image.astype(np.float32) / 255.0 * 2 - 1
                images.append(cfg.fe.trans(img))
                poses.append(obs[cfg.fe.model.hidden_dim :])
                pbar.update(1)

    env.close()

    np.save("data/sim_rgbd_random", images)
    np.save("data/sim_pose_random", poses)


if __name__ == "__main__":
    main()
