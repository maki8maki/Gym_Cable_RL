from tqdm import tqdm
import numpy as np
import pickle
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import gym_cable

from utils import return_transition, check_freq
from config import Config

class CombExecuter:
    def __init__(self, env_name: str, cfg: Config, options=None):
        gym_cable.register_robotics_envs()
        self.env = gym.make(env_name, render_mode="rgb_array", max_episode_steps=cfg.nsteps, is_random=False)
        self.writer = SummaryWriter(log_dir=cfg.output_dir)
        self.act_space = gym.spaces.Box(
            low = self.env.action_space.low[:cfg.rl.act_dim],
            high = self.env.action_space.high[:cfg.rl.act_dim]
        )
        self.cfg = cfg
        self.options = options
        self.update_count = 0
        
        self.cfg.fe.model.load_state_dict(torch.load(f'./model/{self.cfg.fe.model_name}', map_location=self.cfg.device))
        self.cfg.fe.model.eval()
        self.cfg.rl.model.train()
    
    def normalize_state(self, state):
        # 連続値の状態を[-1,1]の範囲に正規化
        normalized_state = {}
        for key in state.keys():
            state_mean = 0.5 * (self.env.observation_space[key].high + self.env.observation_space[key].low)
            state_halfwidth = 0.5 * (self.env.observation_space[key].high - self.env.observation_space[key].low)
            normalized_state[key] = ((state[key].astype(np.float32) - state_mean) / state_halfwidth).astype(np.float32)
        return normalized_state
    
    def obs2state(self, obs, image_list=['rgb_image', 'depth_image']):
        normalized_obs = self.normalize_state(obs)
        image = normalized_obs[image_list[0]]
        for name in image_list[1:]:
            image = np.concatenate([image, normalized_obs[name]], axis=2)
        image = torch.tensor(self.cfg.fe.trans(image), dtype=torch.float, device=self.cfg.device)
        hs = self.cfg.fe.model.forward(image).cpu().squeeze().detach().numpy()
        state = np.concatenate([hs, normalized_obs['observation'][:self.cfg.rl.obs_dim]])
        return state
    
    def reset_get_state(self):
        obs, _ = self.env.reset(options=self.options)
        state = self.obs2state(obs)
        return state
        
    def set_action(self, state: np.ndarray, action: np.ndarray):
        if (self.cfg.rl.act_dim < self.env.action_space.shape[0]):
            action = np.concatenate([action, np.zeros((self.env.action_space.shape[0]-self.cfg.rl.act_dim,))])
        next_obs, reward, terminated, truncated, _ = self.env.step(action)
        next_state = self.obs2state(next_obs)
        transition = return_transition(state, next_state, reward, action, terminated, truncated)
        return transition
    
    def gathering_data(self):
        data_path = os.path.join(os.getcwd(), 'data', self.cfg.buffer_name)
        if self.cfg.gathering_data:
            state = self.reset_get_state()
            for _ in tqdm(range(self.cfg.memory_size)):
                action = self.act_space.sample()
                transition = self.set_action(state, action)
                self.cfg.replay_buffer.append(transition)
                if transition['done']:
                    state = self.reset_get_state()
                else:
                    state = transition['next_state']
            with open(data_path, 'wb') as f:
                pickle.dump(self.cfg.replay_buffer, f)
            # ログ出力フォルダにも保存
            with open(os.path.join(self.cfg.output_dir, self.cfg.buffer_name), 'wb') as f:
                pickle.dump(self.cfg.replay_buffer, f)
        else:
            with open(data_path, 'rb') as f:
                self.cfg.replay_buffer = pickle.load(f)
            # ログ出力フォルダにも保存
            with open(os.path.join(self.cfg.output_dir, self.cfg.buffer_name), 'wb') as f:
                pickle.dump(self.cfg.replay_buffer, f)
    
    def train_step_loop(self, episode: int, state: np.ndarray):
        episode_reward = 0
        for step in range(self.cfg.nsteps):
            action = self.cfg.rl.model.get_action(state)
            transition = self.set_action(state, action)
            self.cfg.replay_buffer.append(transition)
            episode_reward += transition['reward']
            if self.update_count % self.cfg.update_every == 0:
                for upc in range(self.cfg.update_every):
                    batch = self.cfg.replay_buffer.sample(self.cfg.rl.model.batch_size)
                    self.cfg.rl.model.update_from_batch(batch)
                    num = self.update_count - (self.cfg.update_every - upc)
                    for key in self.cfg.rl.model.info.keys():
                        self.writer.add_scalar('train/'+key, self.cfg.rl.model.info[key], num)
            self.update_count += 1
            if transition['done']:
                break
            else:
                state = transition['next_state']
        self.writer.add_scalar('train/reward', episode_reward, episode+1)
        self.writer.add_scalar('train/step', step, episode+1)
     
    def eval_episode_loop(self, episode: int, frames: list, titles: list):
        self.cfg.rl.model.eval()
        eval_reward = 0.0
        steps = 0.0
        for evalepisode in range(self.cfg.nevalepisodes):
            save_frames = (evalepisode==0) and check_freq(self.cfg.nepisodes, episode, self.cfg.save_anim_num)
            state = self.reset_get_state()
            if save_frames:
                frames.append(self.env.render())
                titles.append(f'Episode {episode+1}')
            for step in range(self.cfg.nsteps):
                action = self.cfg.rl.model.get_action(state, deterministic=True)
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
        self.cfg.fe.model.eval()
        self.cfg.rl.model.train()
        for episode in tqdm(range(self.cfg.nepisodes)):
            state = self.reset_get_state()
            self.train_step_loop(episode, state)
            if check_freq(self.cfg.nepisodes, episode, self.cfg.eval_num):
                self.eval_episode_loop(episode, frames, titles)
                self.cfg.rl.model.train()
    
    def test_step_loop(self, frames: list, titles: list):
        self.cfg.rl.model.eval()
        state = self.reset_get_state()
        frames.append(self.env.render())
        titles.append('Step 0')
        for step in range(self.cfg.nsteps):
            action = self.cfg.rl.model.get_action(state, deterministic=True)
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
        

class CLCombExecuter(CombExecuter):
    def __init__(self, env_name: str, cfg: Config, cl_scheduler: list):
        super().__init__(env_name=env_name, cfg=cfg, options=cl_scheduler[0][1])
        self.cl_scheduler = cl_scheduler
    
    def train_epsiode_loop(self, frames: list, titles: list):
        self.cfg.fe.model.eval()
        self.cfg.rl.model.train()
        idx = 0
        for episode in tqdm(range(self.cfg.nepisodes)):
            if self.cl_scheduler[idx][0] <= episode+1:
                self.options = self.cl_scheduler[idx][1]
                idx = min(idx+1, len(self.cl_scheduler)-1)
            state = self.reset_get_state()
            self.train_step_loop(episode, state)
            if check_freq(self.cfg.nepisodes, episode, self.cfg.eval_num):
                self.eval_episode_loop(episode, frames, titles)
                self.cfg.rl.model.train()
