import numpy as np
import gymnasium as gym
import torch
from torchvision.transforms import functional as F
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
from utils import set_seed, obs2state, yes_no_input

if __name__ == '__main__':
    seed = 42
    set_seed(seed)
    
    if not yes_no_input("logdir for tensorboard and filename of model parameter"):
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
    env = gym.make("MZ04CableGrasp-v0", render_mode="rgb_array", max_episode_steps=nsteps)
    config = {
        "image_size": (img_height, img_width, 4),
        "hidden_dim": data["hidden_dim"],
        "lr": data["lr"]
    }
    
    model = DCAE(**config).to(device)
    trans = lambda img: cv2.resize(img, (img_width, img_height))
    
    now = datetime.now()
    writer = SummaryWriter(log_dir='./logs/DCAE/'+now.strftime('%Y%m%d-%H%M'))
    
    batch_size = 128
    memory_size = 10000
    nepochs = 100
    
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
                imgs.append(F.to_tensor(state['image']))
                pbar.update(1)

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
        writer.add_scalar('train/loss', np.mean(train_loss), epoch)
        writer.add_scalar('test/loss', np.mean(test_loss), epoch)
        if (epoch+1) % (nepochs/10) == 0:
            x = random.choice(test_list).to(device)
            _, y = model.forward(x, return_pred=True)
            x = x.squeeze()
            y = y.squeeze()
            writer.add_image('original/rgb/'+str(epoch+1), x[:3], epoch)
            writer.add_image('original/depth/'+str(epoch+1), x[3:], epoch)
            writer.add_image('reconstructed/rgb/'+str(epoch+1), y[:3], epoch)
            writer.add_image('reconstructed/depth/'+str(epoch+1), y[3:], epoch)
    
    torch.save(model.state_dict(), os.path.join(pwd, "model/DCAE.pth"))

    env.close()
    writer.close()