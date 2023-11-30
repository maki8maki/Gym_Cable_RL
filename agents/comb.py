import numpy as np
import torch
import torchvision.transforms.functional as TF
from gymnasium import spaces
from agents.utils import ReplayBuffer
from agents.DCAE import DCAE
from agents.DDPG import DDPG
from agents.SAC import SAC

class Comb:
    def __init__(self, fe, rl, image_size, hidden_dim, observation_space, action_space, 
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
    
    def get_action(self, state):
        image_tensor = TF.to_tensor(state['image']).to(self.device)
        h = self.fe.forward(image_tensor).squeeze().detach() * 2.0 - 1.0
        obs_tensor = torch.tensor(state['observation'], dtype=torch.float, device=self.device)
        _state = torch.cat((h, obs_tensor))
        return self.rl.get_action(_state)
    
    def batch_to_hidden_state(self, batch):
        imgs, rbs, next_imgs, next_rbs = [], [], [], []
        for state, next_state in zip(batch['states'], batch['next_states']):
            imgs.append(state['image'])
            rbs.append(state['observation'])
            next_imgs.append(next_state['image'])
            next_rbs.append(next_state['observation'])
        imgs = torch.tensor(np.array(imgs), dtype=torch.float, device=self.device).permute(0, 3, 1, 2)
        rbs = torch.tensor(np.array(rbs), dtype=torch.float, device=self.device)
        next_imgs = torch.tensor(np.array(next_imgs), dtype=torch.float, device=self.device).permute(0, 3, 1, 2)
        next_rbs = torch.tensor(np.array(next_rbs), dtype=torch.float, device=self.device)
        hs, imgs_pred = self.fe.forward(imgs, return_pred=True)
        hs = hs.detach()
        loss = self.fe.loss_func(imgs_pred, imgs)
        next_hs = self.fe.forward(next_imgs).detach()
        return loss, torch.cat((hs, rbs), axis=1), torch.cat((next_hs, next_rbs), axis=1)
    
    def update(self):
        batch = self.replay_buffer.sample(self.rl.batch_size)
        self.fe.optim.zero_grad()
        loss, batch["states"], batch["next_states"] = self.batch_to_hidden_state(batch)
        loss.backward()
        self.fe.optim.step()
        self.rl.update_from_batch(batch)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        state_dicts = torch.load(path, map_location=self.device)
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

class DCAE_DDPG(Comb):
    def __init__(self, kwargs):
        super().__init__(fe=DCAE, rl=DDPG, **kwargs)

class DCAE_SAC(Comb):
    def __init__(self, kwargs):
        super().__init__(fe=DCAE, rl=SAC, **kwargs)