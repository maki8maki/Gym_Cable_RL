import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import cv2
import json
import os
from absl import logging
from tqdm import tqdm
from datetime import datetime
import random

import gym_cable

from agents.DCAE import DCAE
from agents.utils import SSIMLoss
from utils import set_seed, obs2state, yes_no_input, EarlyStopping

if __name__ == '__main__':
    seed = 42
    set_seed(seed)
    
    if not yes_no_input("logdir for tensorboard and filename of model parameter and data"):
        exit()
    
    pwd = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(pwd, "params.json"), "r") as f:
        data = json.load(f)
    nsteps = data["nsteps"]
    img_width = data["img_width"]
    img_height = data["img_height"]
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        logging.warning("You are using CPU!!")
    
    gym_cable.register_robotics_envs()
    env = gym.make("MZ04CableGrasp-v0", render_mode="rgb_array", max_episode_steps=nsteps, is_random=True)
    config = {
        "image_size": (img_height, img_width, 4),
        "hidden_dim": data["hidden_dim"],
        "lr": data["lr"],
        "net_activation": nn.GELU(),
        "loss_func": SSIMLoss(channel=4),
    }
    
    model = DCAE(**config).to(device)
    trans = lambda img: cv2.resize(img, (img_width, img_height)).transpose(2, 0, 1) * 0.5 + 0.5
    
    now = datetime.now()
    writer = SummaryWriter(log_dir='./logs/DCAE/'+now.strftime('%Y%m%d-%H%M'))
    
    gathering_data = True
    data_path = './data/grasp_r_rgbd_w-init.npy'
    if gathering_data:
        memory_size = 10000
        imgs = []
        obs, info = env.reset(seed=seed)
        with tqdm(total=memory_size) as pbar:
            while len(imgs) < memory_size:
                action = env.action_space.sample()
                next_obs, reward, success, done, info = env.step(action)
                next_img = env.render()
                if success or done:
                    obs, info = env.reset()
                else:
                    # 初期状態以外での画像を集める
                    obs = next_obs
                    state = obs2state(obs, env.observation_space, trans)
                    imgs.append(state['image'])
                    pbar.update(1)
        # for _ in tqdm(range(memory_size)):
        #     state = obs2state(obs, env.observation_space, trans)
        #     imgs.append(state['image'])
        #     action = env.action_space.sample()
        #     next_obs, reward, success, done, info = env.step(action)
        #     next_img = env.render()
        #     if success or done:
        #         obs, info = env.reset()
        #     else:
        #         obs = next_obs
        np.save(data_path, np.array(imgs))
    else:
        imgs = np.load(data_path)
    
    batch_size = 128
    nepochs = 500
    model_path = "model/DCAE_r_gelu_ssim_w-init.pth"
    early_stopping = EarlyStopping(verbose=True, patience=100, path=model_path, trace_func=tqdm.write)
    
    train_imgs, test_imgs = torch.utils.data.random_split(imgs, [0.7, 0.3])
    train_data = torch.utils.data.DataLoader(dataset=train_imgs, batch_size=batch_size, shuffle=True)
    test_data = torch.utils.data.DataLoader(dataset=test_imgs, batch_size=batch_size, shuffle=False)
    test_list = list(test_imgs)
    
    for epoch in tqdm(range(nepochs)):
        train_loss, test_loss = [], []
        model.train()
        for x in train_data:
            x = x.to(device)
            _, y = model.forward(x, return_pred=True)
            loss = model.loss_func(y, x)
            model.optim.zero_grad()
            loss.backward()
            model.optim.step()
            train_loss.append(loss.cpu().detach().numpy())
        
        model.eval()
        with torch.no_grad():
            for x in test_data:
                x = x.to(device)
                _, y = model.forward(x, return_pred=True)
                loss = model.loss_func(y, x)
                test_loss.append(loss.cpu().detach().numpy())
        writer.add_scalar('train/loss', np.mean(train_loss), epoch+1)
        writer.add_scalar('test/loss', np.mean(test_loss), epoch+1)
        if (epoch+1) % (nepochs/10) == 0:
            x = torch.tensor(random.choice(test_list)).to(device)
            _, y = model.forward(x, return_pred=True)
            x = x.squeeze()
            y = y.squeeze()
            writer.add_image('rgb/'+str(epoch+1)+'_original', x[:3], epoch+1)
            writer.add_image('depth/'+str(epoch+1)+'_original', x[3:], epoch+1)
            writer.add_image('rgb/'+str(epoch+1)+'_reconstructed', y[:3], epoch+1)
            writer.add_image('depth/'+str(epoch+1)+'_reconstructed', y[3:], epoch+1)
        if early_stopping(np.mean(test_loss), model):
            break
    
    torch.save(model.state_dict(), os.path.join(pwd, model_path))

    env.close()
    writer.flush()
    writer.close()
