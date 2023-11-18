import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import os
import json
from absl import logging
from tqdm import tqdm
import gymnasium as gym
import gym_cable

from utils import set_seed, anim, obs2state
from agents.comb import DCAE_DDPG

if __name__ == '__main__':
    seed = 42
    set_seed(seed)
    
    pwd = os.path.dirname(os.path.abspath(__file__)) + "/"
    with open(pwd+"params.json", "r") as f:
        data = json.load(f)
    nepisodes = data["nepisodes"]
    nsteps = data["nsteps"]
    memory_size = data["memory_size"]
    img_width = data["img_width"]
    img_height = data["img_height"]
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        logging.warning("You are using cpu!!")
    
    gym_cable.register_robotics_envs()
    env = gym.make("MZ04CableGrasp-v0", render_mode="rgb_array", max_episode_steps=nepisodes)
    config = {
        "image_size": (img_height, img_width, 4),
        "hidden_dim": data["hidden_dim"],
        "observation_space": env.observation_space["observation"],
        "action_space": env.action_space,
        "gamma": data["gamma"],
        "batch_size": data["batch_size"],
        "memory_size": memory_size,
        "device": device
    }
    
    agent = DCAE_DDPG(**config)
    trans = lambda img: cv2.resize(img, (img_width, img_height))

    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    for i in tqdm(range(memory_size)):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        state = obs2state(obs, env.observation_space, trans)
        next_state = obs2state(next_obs, env.observation_space, trans)
        transition = {
            'state': state,
            'next_state': next_state,
            'reward': reward,
            'action': action,
            'success': int(terminated), # エピソードの終了（成功）
            'done': int(truncated) # エピソードの打ち切り（失敗、エピソードの上限に達する）
        }
        agent.ddpg.replay_buffer.append(transition)
        frames.append(env.render())
        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs = next_obs
    # anim(frames)
    
    episode_rewards = []
    num_average_epidodes = 5
    for episode in tqdm(range(nepisodes)):
        obs, _ = env.reset()
        episode_reward = 0
        for t in range(nsteps):
            state = obs2state(obs, env.observation_space, trans)
            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = obs2state(next_obs, env.observation_space, trans)
            transition = {
                'state': state,
                'next_state': next_state,
                'reward': reward,
                'action': action,
                'success': int(terminated),
                'done': int(truncated)
            }
            agent.ddpg.replay_buffer.append(transition)
            episode_reward += reward
            agent.update()
            if terminated or truncated:
                break
            else:
                obs = next_obs
        episode_rewards.append(episode_reward)
        if (episode+1) % 5 == 0:
            print("Episode %d finished | Episode reward %f" % (episode+1, episode_reward))

    # 累積報酬の移動平均を表示
    moving_average = np.convolve(episode_rewards, np.ones(num_average_epidodes)/num_average_epidodes, mode='valid')
    plt.plot(np.arange(len(moving_average)),moving_average)
    plt.title('DDPG: average rewards in %d episodes' % num_average_epidodes)
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.show()
    
    env.close()