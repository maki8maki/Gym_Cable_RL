import numpy as np
from absl import logging
import torch
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.optim as optim

from .buffer import PPOBuffer
from .utils import RL, mlp


class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim]+list(hidden_sizes)+[act_dim], activation)
    
    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)
    
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)
    
    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim]+list(hidden_sizes)+[1], activation)
    
    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()
        self.pi = MLPGaussianActor(obs_dim=obs_dim, act_dim=act_dim, hidden_sizes=hidden_sizes, activation=activation)
        self.v = MLPCritic(obs_dim=obs_dim, hidden_sizes=hidden_sizes, activation=activation)
    
    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()
    
    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            if deterministic:
                a = self.pi.mu_net(state)
            else:
                pi = self.pi._distribution(state)
                a = pi.sample()
        return a.cpu().numpy()

class PPO(RL):
    def __init__(self, obs_dim, act_dim, ac_kwargs=dict(), gamma=0.99, clip_ratio=0.2,
                 pi_lr=1e-3, vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97,
                 steps_per_epoch=4000, target_kl=0.01, device="cpu"):
        super().__init__()
        self.ac = MLPActorCritic(obs_dim=obs_dim, act_dim=act_dim, **ac_kwargs).to(device)
        self.pi_opt = optim.Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_opt = optim.Adam(self.ac.v.parameters(), lr=vf_lr)
        
        self.buffer = PPOBuffer(steps_per_epoch, obs_dim, act_dim, gamma, lam)
        
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.device = device
        
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio*adv, clip_adv)).mean()
        
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info
    
    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs)-ret)**2).mean()
    
    def update_from_batch(self, batch):
        data = self.batch_to_tensor(batch, key_list=['obs', 'act', 'ret', 'adv', 'logp'])
        
        for i in range(self.train_pi_iters):
            self.pi_opt.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                break
            loss_pi.backward()
            self.pi_opt.step()
        for i in range(self.train_v_iters):
            self.vf_opt.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.vf_opt.step()
        self.info['loss_pi'] = loss_pi.to('cpu').detach().numpy().copy().mean()
        self.info['loss_v'] = loss_v.to('cpu').detach().numpy().copy().mean()
    
    def get_action(self, state, deterministic=False):
        state_tensor = super().get_action(state)
        return self.ac.get_action(state_tensor, deterministic=deterministic)
    
    def state_dict(self):
        return self.ac.state_dict()
    
    def load_state_dict(self, state_dict):
        self.ac.load_state_dict(state_dict)
    
    def eval(self):
        self.ac.eval()
    
    def train(self):
        self.ac.train()
    
    def to(self, device):
        super().to(device)
        self.ac.to(device)
