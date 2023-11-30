import copy
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from .utils import mlp

# OpenAI Spinning UPを参考（URL:https://github.com/openai/spinningup.git）

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes, activation, device='cpu'):
        super().__init__()
        self.net = mlp([observation_space.shape[0]] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], action_space.shape[0])
        self.log_std_layer = nn.Linear(hidden_sizes[-1], action_space.shape[0])
        self.action_mean = torch.tensor(0.5*(action_space.high+action_space.low), dtype=torch.float, device=device)
        self.action_halfwidth = torch.tensor(0.5*(action_space.high-action_space.low), dtype=torch.float, device=device)
        
    def forward(self, s, deterministic=False, with_logprob=True):
        net_out = self.net(s)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        
        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = self.action_mean + self.action_halfwidth*torch.tanh(pi_action)
        
        return pi_action, logp_pi
    
class MLPQFunction(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes, activation, device='cpu'):
        super().__init__()
        self.q = mlp([observation_space.shape[0]+action_space.shape[0]] + list(hidden_sizes) + [1], activation)
        self.action_mean = torch.tensor(0.5*(action_space.high+action_space.low), dtype=torch.float, device=device)
        self.action_halfwidth = torch.tensor(0.5*(action_space.high-action_space.low), dtype=torch.float, device=device)
        
    def forward(self, s, a):
        _a = (a-self.action_mean) / self.action_halfwidth
        q = self.q(torch.cat([s, _a], dim=-1))
        return torch.squeeze(q, -1)

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU, device='cpu'):
        super().__init__()
        self.pi = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation, device)
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation, device)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation, device)
    
    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(state, deterministic, False)
            return a.cpu().numpy()

class SAC():
    def __init__(self, observation_space, action_space, ac_kwargs=dict(), gamma=0.99, polyak=0.995,
                 lr=1e-3, alpha=0.2, batch_size=32, device="cpu"):
        self.ac = MLPActorCritic(observation_space, action_space, device=device, **ac_kwargs).to(device)
        self.ac_targ = copy.deepcopy(self.ac)
        self.q_params = itertools.chain(self.ac.q1.parameters(), self.ac.q2.parameters())
        self.pi_opt = optim.Adam(self.ac.pi.parameters(), lr=lr)
        self.q_opt = optim.Adam(self.q_params, lr=lr)
        for p in self.ac_targ.parameters():
            p.requires_grad = False
        self.polyak = polyak
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = device
    
    def batch_to_tensor(self, batch, key_list=['states', 'actions', 'next_states', 'rewards', 'dones']
        ):
        return_list = []
        for key in key_list:
            if isinstance(batch[key], torch.Tensor):
                item = batch[key]
                if item.dtype != torch.float:
                    item = item.to(torch.float)
                if item.device != self.device:
                    item = item.to(self.device)
            else:
                item = torch.tensor(batch[key], dtype=torch.float, device=self.device)
            return_list.append(item)
        return return_list
    
    def compute_loss_q(self, batch):
        states, actions, next_states, rewards, dones = self.batch_to_tensor(batch)
        q1 = self.ac.q1(states, actions)
        q2 = self.ac.q2(states, actions)
        
        with torch.no_grad():
            # Target action
            next_action, logp_na = self.ac.pi(next_states)
            
            # Target Q-Values
            q1_pi_targ = self.ac_targ.q1(next_states, next_action)
            q2_pi_targ = self.ac_targ.q2(next_states, next_action)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            target_q = rewards + self.gamma * (1-dones) * (q_pi_targ-self.alpha*logp_na)
        
        loss_q1 = ((q1-target_q)**2).mean()
        loss_q2 = ((q2-target_q)**2).mean()
        return loss_q1 + loss_q2
    
    def compute_loss_pi(self, batch):
        states = self.batch_to_tensor(batch, ['states'])[0]
        pi, logp_pi = self.ac.pi(states)
        q1_pi = self.ac.q1(states, pi)
        q2_pi = self.ac.q2(states, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        
        return (self.alpha * logp_pi - q_pi).mean()
    
    def update_from_batch(self, batch):
        self.q_opt.zero_grad()
        loss_q = self.compute_loss_q(batch)
        loss_q.backward()
        self.q_opt.step()
        
        for p in self.q_params:
            p.requires_grad = False
        
        self.pi_opt.zero_grad()
        loss_pi = self.compute_loss_pi(batch)
        loss_pi.backward()
        self.pi_opt.step()
        
        for p in self.q_params:
            p.requires_grad = True
            
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1-self.polyak) * p.data)
        
    def get_action(self, state, deterministic=False):
        if isinstance(state, torch.Tensor):
            state_tensor = state
            if state_tensor.dtype != torch.float:
                state_tensor = state_tensor.to(torch.float)
            if state_tensor.device != self.device:
                state_tensor = state_tensor.to(self.device)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
        return self.ac.get_action(state_tensor, deterministic)
    
    def state_dict(self):
        return self.ac.state_dict()
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
    
    def load_state_dict(self, state_dict):
        self.ac.load_state_dict(state_dict)
    
    def eval(self):
        self.ac.eval()