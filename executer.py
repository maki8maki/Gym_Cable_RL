from tqdm import tqdm
import numpy as np
import pickle
import os
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

from agents.utils import RL
from utils import obs2state_through_fe, return_transition, check_freq
from config import Config

class Executer:
    def __init__(self, env: gym.Env, cfg: Config, fe: nn.Module, rl: RL, options=None):
        self.env = env
        self.cfg = cfg
        self.fe = fe
        self.rl = rl
        self.writer = SummaryWriter(log_dir=cfg.output_dir)
        self.options = options
        self.update_count = 0
    
    def reset_get_state(self):
        obs, _ = self.env.reset(options=self.options)
        state = obs2state_through_fe(obs, self.env.observation_space, self.fe, self.cfg.fe.trans, device=self.cfg.device, obs_dim=self.cfg.rl.obs_dim)
        return state
        
    def set_action(self, state: np.ndarray, action: np.ndarray):
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        next_state = obs2state_through_fe(next_obs, self.env.observation_space, self.fe, self.cfg.fe.trans, device=self.cfg.device, obs_dim=self.cfg.rl.obs_dim)
        transition = return_transition(state, next_state, reward, action, terminated, truncated)
        return transition
    
    def gathering_data(self, action_space: gym.spaces.Box):
        data_path = os.path.join(os.getcwd(), 'data', self.cfg.data_name)
        if self.cfg.gathering_data:
            state = self.reset_get_state()
            for _ in tqdm(range(self.cfg.memory_size)):
                action = action_space.sample()
                transition = self.set_action(state, action)
                self.cfg.replay_buffer.append(transition)
                if transition['done']:
                    state = self.reset_get_state()
                else:
                    state = transition['next_state']
            with open(data_path, 'wb') as f:
                pickle.dump(self.cfg.replay_buffer, f)
            # ログ出力フォルダにも保存
            with open(os.path.join(self.cfg.output_dir, self.cfg.data_name), 'wb') as f:
                pickle.dump(self.cfg.replay_buffer, f)
        else:
            with open(data_path, 'rb') as f:
                self.cfg.replay_buffer = pickle.load(f)
    
    def train_step_loop(self, episode: int, state: np.ndarray):
        episode_reward = 0
        for step in range(self.cfg.nsteps):
            action = self.rl.get_action(state)
            transition = self.set_action(state, action)
            self.cfg.replay_buffer.append(transition)
            episode_reward += transition['reward']
            if self.update_count % self.cfg.update_every == 0:
                for upc in range(self.cfg.update_every):
                    batch = self.cfg.replay_buffer.sample(self.cfg.rl.batch_size)
                    self.rl.update_from_batch(batch)
                    num = self.update_count - (self.cfg.update_every - upc)
                    for key in self.rl.info.keys():
                        self.writer.add_scalar('train/'+key, self.rl.info[key], num)
            self.update_count += 1
            if transition['done']:
                break
            else:
                state = transition['next_state']
        self.writer.add_scalar('train/reward', episode_reward, episode+1)
        self.writer.add_scalar('train/step', step, episode+1)
     
    def eval_episode_loop(self, episode: int, frames: list, titles: list):
        self.rl.eval()
        eval_reward = 0.0
        steps = 0.0
        for evalepisode in range(self.cfg.nevalepisodes):
            save_frames = (evalepisode==0) and check_freq(self.cfg.nepisodes, episode, self.cfg.save_anim_num)
            state = self.reset_get_state()
            if save_frames:
                frames.append(self.env.render())
                titles.append(f'Episode {episode+1}')
            for step in range(self.cfg.nsteps):
                action = self.rl.get_action(state, deterministic=True)
                transition = self.set_action(state, action)
                eval_reward += transition['reward']
                if save_frames:
                    frames.append(self.env.render())
                    titles.append(f'Episode {episode+1}')
                if transition['done']:
                    break
                else:
                    state = transition['next_state']
            steps += step + 1.0
        self.writer.add_scalar('test/reward', eval_reward/self.cfg.nevalepisodes, episode+1)
        self.writer.add_scalar('test/step', steps/self.cfg.nevalepisodes, episode+1)

    def train_epsiode_loop(self, frames: list, titles: list):
        self.fe.eval()
        self.rl.train()
        for episode in tqdm(range(self.cfg.nepisodes)):
            state = self.reset_get_state()
            self.train_step_loop(episode, state)
            if check_freq(self.cfg.nepisodes, episode, self.cfg.eval_num):
                self.eval_episode_loop(episode, frames, titles)
                self.rl.train()
    
    def test_step_loop(self, frames: list, titles: list):
        self.rl.eval()
        state = self.reset_get_state()
        frames.append(self.env.render())
        titles.append('Step 0')
        for step in range(self.cfg.nsteps):
            action = self.rl.get_action(state, deterministic=True)
            transition = self.set_action(state, action)
            frames.append(self.env.render())
            titles.append(f'Episode {step+1}')
            if transition['done']:
                break
            else:
                state = transition['next_state']
    
    def close(self):
        self.env.close()
        self.writer.flush()
        self.writer.close()
        

class CLExecuter(Executer):
    def __init__(self, env: gym.Env, cfg: Config, fe: nn.Module, rl: RL, cl_scheduler: list):
        super().__init__(env=env, cfg=cfg, fe=fe, rl=rl, options=cl_scheduler[0][1])
        self.cl_scheduler = cl_scheduler
    
    def train_epsiode_loop(self, frames: list, titles: list):
        self.fe.eval()
        self.rl.train()
        idx = 0
        for episode in tqdm(range(self.cfg.nepisodes)):
            if self.cl_scheduler[idx][0] == episode+1:
                self.options = self.cl_scheduler[idx][1]
                idx = min(idx+1, len(self.cl_scheduler))
            state = self.reset_get_state()
            self.train_step_loop(episode, state)
            if check_freq(self.cfg.nepisodes, episode, self.cfg.eval_num):
                self.eval_episode_loop(episode, frames, titles)
                self.rl.train()
