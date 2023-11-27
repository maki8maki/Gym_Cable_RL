import numpy as np
import torch
import torch.nn.functional as NF
import torch.optim as optim
import torchvision.transforms.functional as TF
from gymnasium import spaces
from agents.DCAE import DCAE
from agents.DDPG import DDPG

class DCAE_DDPG:
    def __init__(self, image_size, hidden_dim, observation_space, action_space, gamma=0.99, lr=1e-3, batch_size=32, memory_size=50000, device="cpu", dcae_loss = NF.mse_loss) -> None:
        self.dcae = DCAE(image_size=image_size, hidden_dim=hidden_dim).to(device)
        self.dcae_optim = optim.Adam(self.dcae.parameters(), lr=lr)
        self.dcae_loss = dcae_loss
        hidden_low = np.full(hidden_dim, -1.0)
        hidden_high = np.full(hidden_dim, 1.0)
        obs_space_low = np.concatenate([hidden_low, observation_space.low])
        obs_space_high = np.concatenate([hidden_high, observation_space.high])
        obs_space = spaces.Box(low=obs_space_low, high=obs_space_high, dtype=np.float64)
        self.ddpg = DDPG(observation_space=obs_space, action_space=action_space, gamma=gamma, lr=lr, batch_size=batch_size, memory_size=memory_size, device=device)
        self.device = device
    
    def get_action(self, state):
        image_tensor = TF.to_tensor(state['image']).to(self.device)
        h = self.dcae.forward(image_tensor).squeeze().detach() * 2.0 - 1.0
        obs_tensor = torch.tensor(state['observation'], dtype=torch.float, device=self.device)
        _state = torch.cat((h, obs_tensor))
        return self.ddpg.get_action(_state)
    
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
        hs, imgs_pred = self.dcae.forward(imgs, return_pred=True)
        hs = hs.detach()
        loss = self.dcae_loss(imgs_pred, imgs)
        next_hs = self.dcae.forward(next_imgs).detach()
        return loss, torch.cat((hs, rbs), axis=1), torch.cat((next_hs, next_rbs), axis=1)

    def update(self):
        batch = self.ddpg.replay_buffer.sample(self.ddpg.batch_size)
        loss, batch["states"], batch["next_states"] = self.batch_to_hidden_state(batch)
        loss.backward()
        self.dcae_optim.step()
        self.dcae_optim.zero_grad()
        self.ddpg.update_from_batch(batch)
    
    def save(self, path):
        torch.save({
            "dcae": self.dcae.state_dict(),
            "actor": self.ddpg.actor.state_dict(),
            "critic": self.ddpg.critic.state_dict()
        }, path)
    
    def load(self, path):
        state_dicts = torch.load(path, map_location=self.device)
        self.dcae.load_state_dict(state_dicts["dcae"])
        self.ddpg.actor.load_state_dict(state_dicts["actor"])
        self.ddpg.critic.load_state_dict(state_dicts["critic"])
    
    def state_dict(self):
        state_dicts = {
            "dcae": self.dcae.state_dict(),
            "actor": self.ddpg.actor.state_dict(),
            "critic": self.ddpg.critic.state_dict()
        }
        return state_dicts
    
    def eval(self):
        self.dcae.eval()
        self.ddpg.actor.eval()
        self.ddpg.critic.eval()