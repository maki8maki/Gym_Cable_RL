import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torchvision.transforms as tf
from omegaconf import OmegaConf

from config import DAConfig


@hydra.main(config_path="conf/train_da", config_name="config", version_base=None)
def main(_cfg: OmegaConf):
    cfg = DAConfig.convert(_cfg)

    model = cfg.model
    model.load(os.path.join("logs", "FE_CycleGAN_VAE", "20241023-1422", "FE_CycleGAN_VAE.pth"))
    model.eval()

    suffix = "_20"

    sim = np.load(f"data/sim{suffix}.npy").astype(np.float32)
    real = np.load(f"data/real{suffix}.npy").astype(np.float32)

    t = tf.Compose([th.tensor])
    with th.no_grad():
        model.set_input({"A": t(sim), "B": t(real)})
        model.forward()

    fake_real: np.ndarray = model.fake_B.cpu().squeeze().numpy().transpose(1, 2, 0) * 0.5 + 0.5
    fake_sim: np.ndarray = model.fake_A.cpu().squeeze().numpy().transpose(1, 2, 0) * 0.5 + 0.5

    sim: np.ndarray = sim.squeeze().transpose(1, 2, 0) * 0.5 + 0.5
    real: np.ndarray = real.squeeze().transpose(1, 2, 0) * 0.5 + 0.5

    d = {
        "sim_rgb": sim[..., :3],
        "sim_depth": sim[..., 3:],
        "fake_real_rgb": fake_real[..., :3],
        "fake_real_depth": fake_real[..., 3:],
        "real_rgb": real[..., :3],
        "real_depth": real[..., 3:],
        "fake_sim_rgb": fake_sim[..., :3],
        "fake_sim_depth": fake_sim[..., 3:],
    }

    for k, v in d.items():
        fig, _ = plt.subplots(figsize=(v.shape[1] / 10, v.shape[0] / 10))
        plt.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.imshow(v, cmap="gray", vmin=0, vmax=1)
        plt.show()
        # plt.savefig(f"logs/{k}{suffix}.png")


if __name__ == "__main__":
    main()
