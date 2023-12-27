import torch
import os
import json
import numpy as np
from datetime import datetime
from absl import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from gymnasium import spaces
import gymnasium as gym
import gym_cable

from utils import set_seed, anim, obs2state, return_transition, yes_no_input
from agents.SAC import SAC
from agents.utils import ReplayBuffer

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
    ntestepisodes = data["ntestepisodes"]
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        logging.warning("You are using CPU!!")
    
    gym_cable.register_robotics_envs()
    env = gym.make("MZ04CableGrasp-v0", render_mode="rgb_array", max_episode_steps=nsteps)
    action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype="float32")
    config = {
        "observation_space": env.observation_space["observation"],
        "action_space": action_space,
        "gamma": data["gamma"],
        "batch_size": data["batch_size"],
        "lr": data["lr"],
        "device": device
    }
    
    agent = SAC(**config)
    replay_buffer = ReplayBuffer(memory_size)
    
    now = datetime.now()
    writer = SummaryWriter(log_dir='./logs/SAC/'+now.strftime('%Y%m%d-%H%M'))

    obs, _ = env.reset(seed=seed)
    ac = np.zeros((3,))
    for _ in tqdm(range(memory_size)):
        action = action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(np.concatenate([action, ac]))
        # next_obs, reward, terminated, truncated, _ = env.step(action)
        state = obs2state(obs, env.observation_space, image_list=[])
        next_state = obs2state(next_obs, env.observation_space, image_list=[])
        transition = return_transition(state, next_state, reward, action, terminated, truncated)
        replay_buffer.append(transition)
        if terminated or truncated:
            obs, info = env.reset()
        else:
            obs = next_obs

    frames = []
    titles = []
    update_every = 50
    update_count = 1
    for episode in tqdm(range(nepisodes)):
        obs, _ = env.reset()
        episode_reward = 0
        for step in range(nsteps):
            state = obs2state(obs, env.observation_space, image_list=[])
            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(np.concatenate([action, ac]))
            # next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state = obs2state(next_obs, env.observation_space, image_list=[])
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
        # if (episode+1) % 50 == 0:
            # Test
            agent.eval()
            test_reward = 0
            steps = 0
            for testepisode in range(ntestepisodes):
                obs, _ = env.reset()
                # if testepisode == 0:
                #     frames.append(env.render())
                #     titles.append("Episode "+str(episode+1))
                for step in range(nsteps):
                    state = obs2state(obs, env.observation_space, image_list=[])
                    action = agent.get_action(state, deterministic=True)
                    next_obs, reward, terminated, truncated, _ = env.step(np.concatenate([action, ac]))
                    # next_obs, reward, terminated, truncated, _ = env.step(action)
                    test_reward += reward
                    # if testepisode == 0:
                    #     frames.append(env.render())
                    #     titles.append("Episode "+str(episode+1))
                    if terminated or truncated:
                        break
                    else:
                        obs = next_obs
                steps += step
            writer.add_scalar('test/reward', test_reward/ntestepisodes, episode+1)
            writer.add_scalar('test/step', steps/ntestepisodes, episode+1)
            agent.train()
    # anim(frames, titles=titles, filename="out/test_only-rl_xy-action1.mp4", show=False)

    agent.eval()
    obs, _ = env.reset()
    frames = [env.render()]
    titles= ["Step "+str(0)]
    for step in range(10):
        state = obs2state(obs, env.observation_space, image_list=[])
        action = agent.get_action(state, deterministic=True)
        next_obs, reward, terminated, truncated, _ = env.step(np.concatenate([action, ac]))
        # next_obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        titles.append("Step "+str(step+1))
        if terminated or truncated:
            break
        else:
            obs = next_obs
    
    anim(frames, titles=titles, filename="out/test_only-rl_xyz-action2.mp4", show=False)
    agent.save("model/test_only-rl_xyz-action.pth")
    
    env.close()
    writer.flush()
    writer.close()