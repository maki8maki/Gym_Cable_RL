import torch
import torch.nn as nn
import os
import json
import cv2
import numpy as np
from absl import logging
from gymnasium import spaces
import gymnasium as gym
import gym_cable

from utils import set_seed, anim, obs2state_through_fe
from agents.SAC import SAC
from agents.DCAE import DCAE
from agents.utils import SSIMLoss

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
    hidden_dim = data["hidden_dim"]
    lr = data["lr"]
    ntestepisodes = data["ntestepisodes"]
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        logging.warning("You are using CPU!!")
    
    gym_cable.register_robotics_envs()
    env = gym.make("MZ04CableGrasp-v0", render_mode="rgb_array", max_episode_steps=nsteps, is_random=False)
    hidden_low = np.full(hidden_dim, -1.0)
    hidden_high = np.full(hidden_dim, 1.0)
    # obs_space_low = np.concatenate([hidden_low, env.observation_space["observation"].low[:3]])
    # obs_space_high = np.concatenate([hidden_high, env.observation_space["observation"].high[:3]])
    obs_space_low = np.concatenate([hidden_low, env.observation_space["observation"].low])
    obs_space_high = np.concatenate([hidden_high, env.observation_space["observation"].high])
    observation_space = spaces.Box(low=obs_space_low, high=obs_space_high, dtype=np.float64)
    action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype="float32")
    rl_config = {
        "observation_space": observation_space,
        "action_space": action_space,
        "gamma": data["gamma"],
        "batch_size": data["batch_size"],
        "lr": lr,
        "device": device
    }
    
    agent = SAC(**rl_config)
    agent.load("./model/test_rl_w-trainedAE_xyz-action.pth")
    agent.eval()
    
    ae_config = {
        "image_size": (img_height, img_width, 4),
        "hidden_dim": hidden_dim,
        "lr": lr,
        "net_activation": nn.GELU(),
        "loss_func": SSIMLoss(channel=4),
    }
    
    model = DCAE(**ae_config).to(device)
    model.load_state_dict(torch.load('./model/DCAE_gelu_ssim_w-init.pth', map_location=device))
    model.eval()
    trans = lambda img: cv2.resize(img, (img_width, img_height)).transpose(2, 0, 1) * 0.5 + 0.5

    obs, _ = env.reset(seed=seed)
    ac = np.zeros((3,))
    
    agent.eval()
    obs, _ = env.reset()
    actions = []
    frames = [env.render()]
    titles = ["Step 0"]
    for step in range(100):
        state = obs2state_through_fe(obs, env.observation_space, model, trans, device=device)
        action = agent.get_action(state, deterministic=True)
        actions.append(action)
        next_obs, reward, terminated, truncated, _ = env.step(np.concatenate([action, ac]))
        frames.append(env.render())
        titles.append("Step "+str(step+1))
        if terminated or truncated:
            break
        else:
            obs = next_obs
    print(obs['observation'])
    print(next_obs['observation'])
    print(truncated)
    # anim(frames, titles=titles, filename="out/rl_w-trainedAE_xyz.mp4", show=False)
    
    env.close()
