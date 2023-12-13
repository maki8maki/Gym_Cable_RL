import matplotlib.pyplot as plt
import torch
import cv2
import os
import json
import numpy as np
from absl import logging
from gymnasium import spaces
import gymnasium as gym
import gym_cable

from utils import set_seed, anim, obs2state
from agents.comb import DCAE_SAC

if __name__ == '__main__':
    seed = 42
    set_seed(seed)
    
    pwd = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(pwd, "params.json"), "r") as f:
        data = json.load(f)
    nepisodes = data["nepisodes"]
    nsteps = data["nsteps"]
    memory_size = data["memory_size"]
    img_width = data["img_width"]
    img_height = data["img_height"]
    ntestepisodes = data["ntestepisodes"]
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        logging.warning("You are using CPU!!")
    
    gym_cable.register_robotics_envs()
    env = gym.make("MZ04CableGrasp-v0", render_mode="rgb_array", max_episode_steps=nepisodes)
    action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype="float32")
    config = {
        "image_size": (img_height, img_width, 4),
        "hidden_dim": data["hidden_dim"],
        "observation_space": env.observation_space["observation"],
        "action_space": action_space,
        "memory_size": memory_size,
        "fe_kwargs": {
            "lr": data["lr"],
        },
        "rl_kwargs": {
            "gamma": data["gamma"],
            "batch_size": data["batch_size"],
            "lr": data["lr"],
        },
        "device": device
    }
    
    agent = DCAE_SAC(config)
    model_path = os.path.join(pwd, "test_1deg-action.pth")
    agent.load(model_path)
    trans = lambda img: cv2.resize(img, (img_width, img_height))

    frames = []
    titles = []
    
    ac = np.zeros((5,))
    agent.eval()
    obs, _ = env.reset()
    terminated, truncated = False, False
    actions = []
    for step in range(100):
        state = obs2state(obs, env.observation_space, trans)
        action = agent.get_action(state, deterministic=True)
        actions.append(action)
        next_obs, reward, terminated, truncated, _ = env.step(np.concatenate([action, ac]))
        if terminated or truncated:
            break
        frames.append(env.render())
        titles.append("Step "+str(step+1))
    
    plt.plot(actions)
    plt.show()
    
    anim(frames, titles=titles)
    
    env.close()