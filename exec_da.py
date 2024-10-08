import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from omegaconf import OmegaConf

from config import DAConfig


@hydra.main(config_path="conf/train_da", config_name="config", version_base=None)
def main(_cfg: OmegaConf):
    cfg = DAConfig.convert(_cfg)

    model = cfg.model
    model.load(os.path.join("logs", "FE_CycleGAN_VAE", "20241007-1347", "FE_CycleGAN_VAE.pth"))
    model.eval()

    sim = np.load("data/sim.npy") * 2 - 1  # [0 1] -> [-1 1]
    real = np.load("data/real.npy") * 2 - 1

    with th.no_grad():
        model.set_input({"A": th.tensor(sim), "B": th.tensor(real)})
        model.forward()

    fake_real = model.fake_B.cpu().squeeze().numpy().transpose(1, 2, 0) * 0.5 + 0.5
    fake_sim = model.fake_A.cpu().squeeze().numpy().transpose(1, 2, 0) * 0.5 + 0.5

    sim = sim.squeeze().transpose(1, 2, 0) * 0.5 + 0.5
    real = real.squeeze().transpose(1, 2, 0) * 0.5 + 0.5

    print(sim.min(), sim.max())
    print(fake_sim.min(), fake_sim.max())
    print(real.min(), real.max())
    print(fake_real.min(), fake_real.max())

    plt.axis("off")
    plt.imshow(sim[..., :3], vmin=0, vmax=1)
    plt.show()

    plt.axis("off")
    plt.imshow(sim[..., 3:], cmap="gray", vmin=0, vmax=1)
    plt.show()

    plt.axis("off")
    plt.imshow(fake_real[..., :3], vmin=0, vmax=1)
    plt.show()

    plt.axis("off")
    plt.imshow(fake_real[..., 3:], cmap="gray", vmin=0, vmax=1)
    plt.show()

    plt.axis("off")
    plt.imshow(real[..., :3], vmin=0, vmax=1)
    plt.show()

    plt.axis("off")
    plt.imshow(real[..., 3:], cmap="gray", vmin=0, vmax=1)
    plt.show()

    plt.axis("off")
    plt.imshow(fake_sim[..., :3], vmin=0, vmax=1)
    plt.show()

    plt.axis("off")
    plt.imshow(fake_sim[..., 3:], cmap="gray", vmin=0, vmax=1)
    plt.show()


if __name__ == "__main__":
    main()
