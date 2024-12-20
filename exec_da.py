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

    sim = np.load("data/sim.npy").astype(np.float32)
    real = np.load("data/real.npy").astype(np.float32)

    t = tf.Compose([th.tensor])
    with th.no_grad():
        model.set_input({"A": t(sim), "B": t(real)})
        model.forward()

    fake_real = model.fake_B.cpu().squeeze().numpy().transpose(1, 2, 0) * 0.5 + 0.5
    fake_sim = model.fake_A.cpu().squeeze().numpy().transpose(1, 2, 0) * 0.5 + 0.5

    sim = sim.squeeze().transpose(1, 2, 0) * 0.5 + 0.5
    real = real.squeeze().transpose(1, 2, 0) * 0.5 + 0.5

    fig, _ = plt.subplots(figsize=(sim.shape[1] / 10, sim.shape[0] / 10))
    plt.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(sim[..., :3], vmin=0, vmax=1)
    plt.show()

    fig, _ = plt.subplots(figsize=(sim.shape[1] / 10, sim.shape[0] / 10))
    plt.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(sim[..., 3:], cmap="gray", vmin=0, vmax=1)
    plt.show()

    fig, _ = plt.subplots(figsize=(sim.shape[1] / 10, sim.shape[0] / 10))
    plt.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(fake_real[..., :3], vmin=0, vmax=1)
    plt.show()

    fig, _ = plt.subplots(figsize=(sim.shape[1] / 10, sim.shape[0] / 10))
    plt.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(fake_real[..., 3:], cmap="gray", vmin=0, vmax=1)
    plt.show()

    fig, _ = plt.subplots(figsize=(sim.shape[1] / 10, sim.shape[0] / 10))
    plt.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(real[..., :3], vmin=0, vmax=1)
    plt.show()

    fig, _ = plt.subplots(figsize=(sim.shape[1] / 10, sim.shape[0] / 10))
    plt.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(real[..., 3:], cmap="gray", vmin=0, vmax=1)
    plt.show()

    fig, _ = plt.subplots(figsize=(sim.shape[1] / 10, sim.shape[0] / 10))
    plt.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(fake_sim[..., :3], vmin=0, vmax=1)
    plt.show()

    fig, _ = plt.subplots(figsize=(sim.shape[1] / 10, sim.shape[0] / 10))
    plt.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.imshow(fake_sim[..., 3:], cmap="gray", vmin=0, vmax=1)
    plt.show()


if __name__ == "__main__":
    main()
