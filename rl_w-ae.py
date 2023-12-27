import torch
import torch.nn as nn
import os
import json
import cv2
import numpy as np
from datetime import datetime
from absl import logging
from tqdm import tqdm
import pickle
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces
import gymnasium as gym
import gym_cable

from utils import set_seed, anim, obs2state_through_fe, return_transition, yes_no_input
from agents.SAC import SAC
from agents.DCAE import DCAE
from agents.utils import ReplayBuffer, SSIMLoss

if __name__ == '__main__':
    seed = 42
    set_seed(seed)
    
    if not yes_no_input("logdir for tensorboard and filename of animation and model parameters"):
        exit()
    
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
    env = gym.make("MZ04CableGrasp-v0", render_mode="rgb_array", max_episode_steps=nsteps)
    hidden_low = np.full(hidden_dim, -1.0)
    hidden_high = np.full(hidden_dim, 1.0)
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
    replay_buffer = ReplayBuffer(memory_size)
    
    ae_config = {
        "image_size": (img_height, img_width, 4),
        "hidden_dim": hidden_dim,
        "lr": lr,
        "net_activation": nn.GELU(),
        "loss_func": SSIMLoss(channel=4),
    }
    
    model = DCAE(**ae_config).to(device)
    model.load_state_dict(torch.load('./model/DCAE_gelu_ssim_with-init_best.pth', map_location=device))
    model.eval()
    trans = lambda img: cv2.resize(img, (img_width, img_height)).transpose(2, 0, 1) * 0.5 + 0.5
    
    now = datetime.now()
    writer = SummaryWriter(log_dir='./logs/SAC_w-TrainedDCAE/'+now.strftime('%Y%m%d-%H%M'))

    obs, _ = env.reset(seed=seed)
    ac = np.zeros((3,))
    
    gathering_data = True
    data_path = f'./data/buffer_w-hidden-state_{action_space.shape[0]}_{memory_size}.pcl'
    if gathering_data:
        for _ in tqdm(range(memory_size)):
            action = action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(np.concatenate([action, ac]))
            # next_obs, reward, terminated, truncated, _ = env.step(action)
            state = obs2state_through_fe(obs, env.observation_space, model, trans, device=device)
            next_state = obs2state_through_fe(next_obs, env.observation_space, model, trans, device=device)
            transition = return_transition(state, next_state, reward, action, terminated, truncated)
            replay_buffer.append(transition)
            if terminated or truncated:
                obs, info = env.reset()
            else:
                obs = next_obs
        with open(data_path, 'wb') as f:
            pickle.dump(replay_buffer, f)
    else:
        with open(data_path, 'rb') as f:
            replay_buffer = pickle.load(f)

    frames = []
    titles = []
    update_every = 50
    update_count = 1
    for episode in tqdm(range(nepisodes)):
        obs, _ = env.reset()
        episode_reward = 0
        for step in range(nsteps):
            state = obs2state_through_fe(obs, env.observation_space, model, trans, device=device)
            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(np.concatenate([action, ac]))
            # next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = obs2state_through_fe(next_obs, env.observation_space, model, trans, device=device)
            transition = return_transition(state, next_state, reward, action, terminated, truncated)
            replay_buffer.append(transition)
            episode_reward += reward
            if update_count % update_every == 0:
                for upc in range(update_every):
                    batch = replay_buffer.sample(agent.batch_size)
                    agent.update_from_batch(batch)
                    num = update_count - (update_every - upc)
                    for key in agent.info.keys():
                        writer.add_scalar('train/'+key, agent.info[key], num)
            update_count += 1
            if terminated or truncated:
                break
            else:
                obs = next_obs
        writer.add_scalar('train/reward', episode_reward, episode+1)
        writer.add_scalar('train/step', step, episode+1)
        if (episode+1) % (nepisodes/10) == 0:
            tqdm.write("Episode %d finished when step %d | Episode reward %f" % (episode+1, step+1, episode_reward))
        if (episode+1) % (nepisodes/50) == 0:
            # Test
            agent.eval()
            test_reward = 0
            steps = 0
            for testepisode in range(ntestepisodes):
                obs, _ = env.reset()
                if testepisode == 0 and (episode+1) % (nepisodes/10) == 0:
                    frames.append(env.render())
                    titles.append("Episode "+str(episode+1))
                for step in range(nsteps):
                    state = obs2state_through_fe(obs, env.observation_space, model, trans, device=device)
                    action = agent.get_action(state, deterministic=True)
                    next_obs, reward, terminated, truncated, _ = env.step(np.concatenate([action, ac]))
                    # next_obs, reward, terminated, truncated, _ = env.step(action)
                    test_reward += reward
                    if testepisode == 0 and (episode+1) % (nepisodes/10) == 0:
                        frames.append(env.render())
                        titles.append("Episode "+str(episode+1))
                    if terminated or truncated:
                        break
                    else:
                        obs = next_obs
                steps += step
            writer.add_scalar('test/reward', test_reward/ntestepisodes, episode+1)
            writer.add_scalar('test/step', steps/ntestepisodes, episode+1)
            agent.train()
    anim(frames, titles=titles, filename="out/test_rl_w-trainedAE_xyz-action1.mp4", show=False)

    agent.eval()
    obs, _ = env.reset()
    frames = [env.render()]
    titles= ["Step "+str(0)]
    for step in range(10):
        state = obs2state_through_fe(obs, env.observation_space, model, trans, device=device)
        action = agent.get_action(state, deterministic=True)
        next_obs, reward, terminated, truncated, _ = env.step(np.concatenate([action, ac]))
        # next_obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        titles.append("Step "+str(step+1))
        if terminated or truncated:
            break
        else:
            obs = next_obs
    
    anim(frames, titles=titles, filename="out/test_rl_w-trainedAE_xyz-action2.mp4", show=False)
    agent.save("model/test_rl_w-trainedAE_xyz-action.pth")
    
    env.close()
    writer.flush()
    writer.close()