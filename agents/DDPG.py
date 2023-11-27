import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.utils import ReplayBuffer

class ActorNetwork(nn.Module):
    def __init__(self, num_state, action_space, hidden_size=16, device="cpu"):
        super(ActorNetwork, self).__init__()
        self.action_mean = torch.tensor(0.5*(action_space.high+action_space.low), dtype=torch.float, device=device)
        self.action_halfwidth = torch.tensor(0.5*(action_space.high-action_space.low), dtype=torch.float, device=device)
        self.fc1 = nn.Linear(num_state, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space.shape[0])

    def forward(self, s):
        h = F.elu(self.fc1(s))
        h = F.elu(self.fc2(h))
        a = self.action_mean + self.action_halfwidth*torch.tanh(self.fc3(h))
        return a

class CriticNetwork(nn.Module):
    def __init__(self, num_state, action_space, hidden_size=16, device="cpu"):
        super(CriticNetwork, self).__init__()
        self.action_mean = torch.tensor(0.5*(action_space.high+action_space.low), dtype=torch.float, device=device)
        self.action_halfwidth = torch.tensor(0.5*(action_space.high-action_space.low), dtype=torch.float, device=device)
        self.fc1 = nn.Linear(num_state+action_space.shape[0], hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, s, a):
        a = (a-self.action_mean)/self.action_halfwidth
        h = F.elu(self.fc1(torch.cat([s,a],1)))
        h = F.elu(self.fc2(h))
        q = self.fc3(h)
        return q

class DDPG:
    def __init__(self, observation_space, action_space, gamma=0.99, polyak=0.995,
                 lr=1e-3, batch_size=32, memory_size=50000, act_noise=0.1, device="cpu"):
        self.num_state = observation_space.shape[0]
        self.num_action = action_space.shape[0]
        self.acion_space = action_space
        self.gamma = gamma  # 割引率
        self.polyak = polyak
        self.batch_size = batch_size
        self.act_noise = act_noise
        self.actor = ActorNetwork(self.num_state, action_space, device=device).to(device)
        self.actor_target = copy.deepcopy(self.actor)  # actorのターゲットネットワーク
        for p in self.actor_target.parameters():
            p.requires_grad = False
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = CriticNetwork(self.num_state, action_space, device=device).to(device)
        self.critic_target = copy.deepcopy(self.critic)  # criticのターゲットネットワーク
        for p in self.critic_target.parameters():
            p.requires_grad = False
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(memory_size)
        self.device = device

    # リプレイバッファからサンプルされたミニバッチをtensorに変換
    def batch_to_tensor(self, batch):
        key_list = ['states', 'actions', 'next_states', 'rewards', 'dones']
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

    # actorとcriticを更新
    def update(self):
        batch = self.replay_buffer.sample(self.batch_size)
        self.update_from_batch(batch)
    
    def update_from_batch(self, batch):
        states, actions, next_states, rewards, dones = self.batch_to_tensor(batch)
        # criticの更新
        self.critic_optimizer.zero_grad()
        with torch.no_grad():
            target_q = (rewards + self.gamma * (1-dones) * self.critic_target(next_states, self.actor_target(next_states)).squeeze()).data
        q = self.critic(states, actions).squeeze()
        critic_loss = F.mse_loss(q, target_q)
        critic_loss.backward()
        self.critic_optimizer.step()
        # actorの更新
        for p in self.critic.parameters():
            p.requires_grad = False
        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        for p in self.critic.parameters():
            p.requires_grad = True
        # ターゲットネットワークのパラメータを更新
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1-self.polyak) * p.data)
            for p, p_targ in zip(self.actor.parameters(), self.actor_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1-self.polyak) * p.data)

    # Q値が最大の行動を選択
    def get_action(self, state):
        if isinstance(state, torch.Tensor):
            state_tensor = state
            if state_tensor.dtype != torch.float:
                state_tensor = state_tensor.to(torch.float)
            if state_tensor.device != self.device:
                state_tensor = state_tensor.to(self.device)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
           action =  self.actor(state_tensor.view(-1, self.num_state)).view(self.num_action).cpu().numpy()
        action += self.act_noise * np.random.randn(self.num_action)
        return np.clip(action, self.acion_space.low, self.acion_space.high)
