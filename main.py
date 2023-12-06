import matplotlib.pyplot as plt
import torch
import cv2
import os
import json
from absl import logging
from tqdm import tqdm
import gymnasium as gym
import gym_cable

from utils import *
from agents.comb import *

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
    config = {
        "image_size": (img_height, img_width, 4),
        "hidden_dim": data["hidden_dim"],
        "observation_space": env.observation_space["observation"],
        "action_space": env.action_space,
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
    trans = lambda img: cv2.resize(img, (img_width, img_height))

    obs, _ = env.reset(seed=seed)
    # frames = [env.render()]
    for _ in tqdm(range(memory_size)):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        state = obs2state(obs, env.observation_space, trans)
        next_state = obs2state(next_obs, env.observation_space, trans)
        transition = return_transition(state, next_state, reward, action, terminated, truncated)
        agent.replay_buffer.append(transition)
        # frames.append(env.render())
        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs = next_obs

    episode_rewards = []
    test_episodes = []
    test_rewards = []
    frames = []
    titles = []
    update_every = 50
    update_count = 1
    for episode in tqdm(range(nepisodes)):
        obs, _ = env.reset()
        episode_reward = 0
        for step in range(nsteps):
            state = obs2state(obs, env.observation_space, trans)
            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = obs2state(next_obs, env.observation_space, trans)
            transition = return_transition(state, next_state, reward, action, terminated, truncated)
            agent.replay_buffer.append(transition)
            episode_reward += reward
            if update_count % update_every == 0:
                for _ in range(update_every):
                    agent.update()
            update_count += 1
            # if episode % 100 == 0:
            #     frames.append(env.render())
            #     titles.append("Episode "+str(episode+1))
            if terminated or truncated:
                break
            else:
                obs = next_obs
        episode_rewards.append(episode_reward)
        if (episode+1) % (nepisodes/10) == 0:
            tqdm.write("Episode %d finished when step %d | Episode reward %f" % (episode+1, step+1, episode_reward))
        if (episode+1) % (nepisodes/50) == 0:
            # Test
            agent.eval()
            test_reward = 0
            for testepisode in range(ntestepisodes):
                obs, _ = env.reset()
                terminated, truncated = False, False
                while not(terminated or truncated):
                    action = agent.get_action(state, deterministic=True)
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    test_reward += reward
            test_episodes.append(episode)
            test_rewards.append(test_reward/ntestepisodes)
            agent.train()
    
    agent.eval()
    for testepisode in range(ntestepisodes):
        obs, _ = env.reset()
        terminated, truncated = False, False
        while not(terminated or truncated):
            action = agent.get_action(state, deterministic=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            frames.append(env.render())
            titles.append("Episode "+str(testepisode+1))

    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('episode')
    plt.ylabel('rewards')
    # plt.savefig('ep_r.pdf')
    plt.show()
    
    plt.plot(test_episodes, test_rewards)
    plt.title('Test Rewards')
    plt.xlabel('episode')
    plt.ylabel('rewards')
    # plt.savefig('te_r.pdf')
    plt.show()
    
    anim(frames, titles=titles)
    # agent.save("dmil.pth")
    
    env.close()