import numpy as np
from typing import Type
import torch as th
import torchvision.transforms.functional as F
from gymnasium import spaces

from .buffer import ReplayBuffer
from .DCAE import DCAE
from .DDPG import DDPG
from .SAC import SAC
from .utils import FE, RL

class Comb:
    def __init__(self, fe: Type[FE], rl: Type[RL], image_size, hidden_dim, observation_space, action_space, 
                 fe_kwargs=dict(), rl_kwargs=dict(), memory_size=50000, device="cpu"):
        self.fe = fe(image_size=image_size, hidden_dim=hidden_dim, **fe_kwargs).to(device)
        hidden_low = np.full(hidden_dim, -1.0)
        hidden_high = np.full(hidden_dim, 1.0)
        obs_space_low = np.concatenate([hidden_low, observation_space.low])
        obs_space_high = np.concatenate([hidden_high, observation_space.high])
        obs_space = spaces.Box(low=obs_space_low, high=obs_space_high, dtype=np.float64)
        self.rl = rl(observation_space=obs_space, action_space=action_space, device=device, **rl_kwargs)
        self.replay_buffer = ReplayBuffer(memory_size)
        self.device = device
    
    def get_action(self, state, deterministic=False):
        image_tensor = F.to_tensor(state['image']).to(self.device)
        h = self.fe.forward(image_tensor).squeeze().detach()
        obs_tensor = th.tensor(state['observation'], dtype=th.float, device=self.device)
        _state = th.cat((h, obs_tensor))
        return self.rl.get_action(_state, deterministic)
    
    def batch_to_hidden_state(self, batch):
        imgs, rbs, next_imgs, next_rbs = [], [], [], []
        for state, next_state in zip(batch['states'], batch['next_states']):
            imgs.append(state['image'])
            rbs.append(state['observation'])
            next_imgs.append(next_state['image'])
            next_rbs.append(next_state['observation'])
        imgs = th.tensor(np.array(imgs), dtype=th.float, device=self.device).permute(0, 3, 1, 2)
        rbs = th.tensor(np.array(rbs), dtype=th.float, device=self.device)
        next_imgs = th.tensor(np.array(next_imgs), dtype=th.float, device=self.device).permute(0, 3, 1, 2)
        next_rbs = th.tensor(np.array(next_rbs), dtype=th.float, device=self.device)
        hs, imgs_pred = self.fe.forward(imgs, return_pred=True)
        hs = hs.detach()
        loss = self.fe.loss_func(imgs_pred, imgs)
        next_hs = self.fe.forward(next_imgs).detach()
        return loss, th.cat((hs, rbs), axis=1), th.cat((next_hs, next_rbs), axis=1)
    
    def update(self):
        batch = self.replay_buffer.sample(self.rl.batch_size)
        self.fe.optim.zero_grad()
        loss, batch["states"], batch["next_states"] = self.batch_to_hidden_state(batch)
        loss.backward()
        self.fe.optim.step()
        self.rl.update_from_batch(batch)
    
    def save(self, path):
        th.save(self.state_dict(), path)
    
    def load(self, path):
        state_dicts = th.load(path, map_location=self.device)
        self.load_state_dict(state_dicts)
    
    def load_state_dict(self, state_dicts):
        self.fe.load_state_dict(state_dicts[self.fe.__class__.__name__])
        self.rl.load_state_dict(state_dicts[self.rl.__class__.__name__])
    
    def state_dict(self):
        state_dicts = {
            self.fe.__class__.__name__: self.fe.state_dict(),
            self.rl.__class__.__name__: self.rl.state_dict(),
        }
        return state_dicts
    
    def eval(self):
        self.fe.eval()
        self.rl.eval()
    
    def train(self):
        self.fe.train()
        self.rl.train()

class DCAE_DDPG(Comb):
    def __init__(self, kwargs):
        super().__init__(fe=DCAE, rl=DDPG, **kwargs)

class DCAE_SAC(Comb):
    def __init__(self, kwargs):
        super().__init__(fe=DCAE, rl=SAC, **kwargs)