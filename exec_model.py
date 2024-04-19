import json
import os

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from absl import logging

import gym_cable
from agents.DCAE import DCAE
from agents.utils import SSIMLoss
from utils import obs2state, set_seed

if __name__ == "__main__":
    seed = 42
    set_seed(seed)

    pwd = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(pwd, "params.json"), "r") as f:
        data = json.load(f)
    nsteps = data["nsteps"]
    img_width = data["img_width"]
    img_height = data["img_height"]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        logging.warning("You are using CPU!!")

    gym_cable.register_robotics_envs()
    env = gym.make("MZ04CableGrasp-v0", render_mode="rgb_array", max_episode_steps=nsteps, is_random=False)
    config = {
        "image_size": (img_height, img_width, 4),
        "hidden_dim": data["hidden_dim"],
        "lr": data["lr"],
        "net_activation": nn.GELU(),
        "loss_func": SSIMLoss(channel=4),
    }

    model = DCAE(**config).to(device)

    def trans(img):
        return cv2.resize(img, (img_width, img_height)).transpose(2, 0, 1) * 0.5 + 0.5

    batch_size = 128
    nepochs = 500
    model_path = "model/DCAE_gelu_ssim_w-init.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = env.reset()
    state = obs2state(obs, env.observation_space, trans)
    x = torch.tensor(state["image"]).to(device)
    _, y = model.forward(x, return_pred=True)
    x = x.cpu().squeeze().detach().numpy().transpose(1, 2, 0)
    y = y.cpu().squeeze().detach().numpy().transpose(1, 2, 0)

    plt.axis("off")
    plt.imshow(x[..., :3])
    plt.show()

    plt.axis("off")
    plt.imshow(x[..., 3:], cmap="gray")
    plt.show()

    plt.axis("off")
    plt.imshow(y[..., :3])
    plt.show()

    plt.axis("off")
    plt.imshow(y[..., 3:], cmap="gray")
    plt.show()

    env.close()
