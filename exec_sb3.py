import os

import hydra
import numpy as np
import torch as th
from omegaconf import OmegaConf

from config import SB3Config
from utils import anim, set_seed


@hydra.main(config_path="conf/", config_name="sb3", version_base=None)
def main(_cfg: OmegaConf):
    cfg = SB3Config.convert(_cfg)
    set_seed(0)

    model = cfg.model.load(os.path.join("logs", "SB3_SAC_trainedVAE", "20240805-2119", f"{cfg.basename}.zip"))
    policy = model.policy
    policy.set_training_mode(False)

    env = cfg.env

    frames = []
    titles = []
    success_num = 0
    for _ in range(10):
        obs, _ = env.reset()
        frames.append(env.render())
        titles.append("Step 0")
        success_num = 0
        for step in range(cfg.nsteps):
            obs = th.tensor(obs)
            action, _ = policy.predict(obs, deterministic=True)
            next_obs, _, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            titles.append(f"Step {step+1}")
            print(info["is_success"], np.linalg.norm(action) / np.sqrt(6))
            if info["is_success"]:
                success_num += 1
            if terminated or truncated:
                if terminated:
                    success_num += 1
                break
            else:
                obs = next_obs
    print(success_num)
    anim(frames, titles=titles, filename=os.path.join(cfg.output_dir, f"{cfg.basename}.mp4"), show=False, interval=500)

    env.close()


if __name__ == "__main__":
    main()
