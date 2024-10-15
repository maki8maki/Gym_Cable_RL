import os
import sys
from typing import Union

import gymnasium as gym
import numpy as np
import torch as th
import torch.utils.data as th_data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agents.buffer import PrioritizedReplayBuffer
from agents.dataset import UnalignedDataset
from agents.utils import Transition
from config import MODEL_DIR, CombConfig, DAConfig, TrainFEConfig
from utils import EarlyStopping, anim, check_freq


class Executer:
    env: gym.Env
    cfg: Union[CombConfig, TrainFEConfig]
    writer: SummaryWriter

    def close(self):
        self.env.close()
        self.writer.flush()
        self.writer.close()


class FEExecuter(Executer):
    def __init__(self, cfg: TrainFEConfig):
        super().__init__()

        self.env = cfg.env
        self.writer = SummaryWriter(log_dir=cfg.output_dir)
        self.cfg = cfg
        model_path1 = os.path.join(MODEL_DIR, cfg.fe.model_name)
        model_path2 = os.path.join(cfg.output_dir, cfg.fe.model_name)
        self.es = EarlyStopping(patience=self.cfg.es_patience, paths=[model_path1, model_path2], trace_func=tqdm.write)
        self.cfg.fe.model.train()

    def normalize_state(self, state):
        # 連続値の状態を[-1,1]の範囲に正規化
        normalized_state = {}
        for key in state.keys():
            state_mean = 0.5 * (self.env.observation_space[key].high + self.env.observation_space[key].low)
            state_halfwidth = 0.5 * (self.env.observation_space[key].high - self.env.observation_space[key].low)
            normalized_state[key] = ((state[key].astype(np.float32) - state_mean) / state_halfwidth).astype(np.float32)
        return normalized_state

    def obs2state(self, obs, image_list=["rgb_image", "depth_image"]):
        normalized_obs = self.normalize_state(obs)
        if len(image_list) > 0:
            image = normalized_obs[image_list[0]]
            for name in image_list[1:]:
                image = np.concatenate([image, normalized_obs[name]], axis=2)
            state = {"observation": normalized_obs["observation"], "image": self.cfg.fe.trans(image)}
            return state
        else:
            return normalized_obs["observation"]

    def reset_get_state(self):
        obs, _ = self.env.reset()
        state = self.obs2state(obs)
        return state

    def gathering_data(self):
        data_path = os.path.join(os.getcwd(), "data", self.cfg.data_name)
        if self.cfg.gathering_data:
            imgs = []
            state = self.reset_get_state()
            if self.cfg.with_init:
                for _ in tqdm(range(self.cfg.data_size)):
                    imgs.append(state["image"])
                    action = self.env.action_space.sample()
                    next_obs, _, terminated, truncated, _ = self.env.step(action)
                    if terminated or truncated:
                        state = self.reset_get_state()
                    else:
                        state = self.obs2state(next_obs)
            else:
                with tqdm(total=self.cfg.data_size) as pbar:
                    while len(imgs) < self.cfg.data_size:
                        action = self.env.action_space.sample()
                        next_obs, _, terminated, truncated, _ = self.env.step(action)
                        if terminated or truncated:
                            state = self.reset_get_state()
                        else:
                            state = self.obs2state(next_obs)
                            imgs.append(state["image"])
                            pbar.update(1)
            self.imgs = np.array(imgs)
            np.save(data_path, self.imgs)
            np.save(os.path.join(self.cfg.output_dir, self.cfg.data_name), self.imgs)
        else:
            self.imgs = np.load(data_path)
            np.save(os.path.join(self.cfg.output_dir, self.cfg.data_name), self.imgs)

    def train(self):
        train_imgs, test_imgs = th_data.random_split(self.imgs, [0.7, 0.3])
        train_data = th_data.DataLoader(dataset=train_imgs, batch_size=self.cfg.batch_size, shuffle=True)
        test_data = th_data.DataLoader(dataset=test_imgs, batch_size=self.cfg.batch_size, shuffle=False)
        state = self.reset_get_state()
        test_x = th.tensor(state["image"]).to(self.cfg.device)

        for epoch in tqdm(range(1, self.cfg.nepochs + 1)):
            train_loss = []
            self.cfg.fe.model.train()
            for x in train_data:
                x = x.to(self.cfg.device)
                loss = self.cfg.fe.model.loss(x)
                self.cfg.fe.model.optim.zero_grad()
                loss.backward()
                self.cfg.fe.model.optim.step()
                train_loss.append(loss.cpu().detach().numpy())
            train_loss = np.mean(train_loss)

            test_loss = []
            self.cfg.fe.model.eval()
            with th.no_grad():
                for x in test_data:
                    x = x.to(self.cfg.device)
                    loss = self.cfg.fe.model.loss(x)
                    test_loss.append(loss.cpu().detach().numpy())
            test_loss = np.mean(test_loss)
            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("test/loss", test_loss, epoch)
            if check_freq(self.cfg.nepochs, epoch, self.cfg.save_recimg_num):
                y = self.test(test_x)
                self.writer.add_image("rgb/original", test_x[:3] * 0.5 + 0.5, epoch)
                self.writer.add_image("depth/original", test_x[3:] * 0.5 + 0.5, epoch)
                self.writer.add_image("rgb/reconstructed", y[:3] * 0.5 + 0.5, epoch)
                self.writer.add_image("depth/reconstructed", y[3:] * 0.5 + 0.5, epoch)
            if self.es(test_loss, self.cfg.fe.model):
                break

    def test(self, x: th.Tensor):
        test_x = x.unsqueeze(0)
        _, y = self.cfg.fe.model.forward(test_x, return_pred=True)
        y = y.squeeze()
        return y

    def __call__(self):
        self.gathering_data()
        try:
            self.train()
        except KeyboardInterrupt:
            pass
        self.close()


class CombExecuter(Executer):
    def __init__(self, cfg: CombConfig):
        super().__init__()

        self.env = cfg.env
        self.writer = SummaryWriter(log_dir=cfg.output_dir)
        self.cfg = cfg
        self.update_count = 1

        self.cfg.fe.model.eval()
        self.cfg.model.train()

    def set_action(self, state: np.ndarray, action: np.ndarray) -> Transition:
        next_state, reward, terminated, truncated, info = self.env.step(action)
        transition = Transition(state, next_state, reward, action, terminated, truncated, info)
        return transition

    def train_loop(self, frames: list, titles: list):
        state, _ = self.env.reset()
        ep_rew, ep_len = 0, 0
        for step in tqdm(range(1, self.cfg.total_steps + 1)):
            if step >= self.cfg.start_steps:
                action = self.cfg.model.get_action(state)
            else:
                action = self.env.action_space.sample()
            transition = self.set_action(state, action)

            ep_rew += transition.reward
            ep_len += 1
            is_timeout = ep_len == self.cfg.nsteps
            transition.done = 0 if is_timeout else transition.done

            self.cfg.replay_buffer.append(transition)

            if transition.done or is_timeout:
                self.writer.add_scalar("train/episode_reward", ep_rew, step)
                self.writer.add_scalar("train/episode_length", ep_len, step)
                state, _ = self.env.reset()
                ep_rew, ep_len = 0, 0
            else:
                state = transition.next_state

            if step >= self.cfg.update_after and step % self.cfg.update_every == 0:
                for upc in range(self.cfg.update_every):
                    num = step - (self.cfg.update_every - upc)
                    if isinstance(self.cfg.replay_buffer, PrioritizedReplayBuffer):
                        batch = self.cfg.replay_buffer.sample(self.cfg.batch_size, num)
                        loss = self.cfg.model.update_from_batch(batch)
                        self.cfg.replay_buffer.update_priority(loss.flatten())
                    else:
                        batch = self.cfg.replay_buffer.sample(self.cfg.batch_size)
                        _ = self.cfg.model.update_from_batch(batch)
                    for key in self.cfg.model.info.keys():
                        self.writer.add_scalar("train/" + key, self.cfg.model.info[key], num)

            if check_freq(self.cfg.total_steps, step, self.cfg.eval_num):
                self.eval_loop(step, frames, titles)
                self.cfg.model.train()

    def eval_loop(self, cur_step: int, frames: list, titles: list):
        self.cfg.model.eval()
        eval_reward = 0.0
        steps = 0.0
        success_num = 0.0
        for evalepisode in range(self.cfg.nevalepisodes):
            save_frames = evalepisode == 0
            state, _ = self.env.reset()
            if save_frames:
                frames.append(self.env.render())
                titles.append(f"Step {cur_step+1}")
            for step in range(self.cfg.nsteps):
                action = self.cfg.model.get_action(state, deterministic=True)
                transition = self.set_action(state, action)
                eval_reward += transition.reward
                if save_frames:
                    frames.append(self.env.render())
                    titles.append(f"Step {cur_step+1}")
                if transition.done:
                    if transition.success:
                        success_num += 1.0
                    break
                else:
                    state = transition.next_state
            steps += step + 1.0
        self.writer.add_scalar("test/episode_reward", eval_reward / self.cfg.nevalepisodes, cur_step)
        self.writer.add_scalar("test/episode_length", steps / self.cfg.nevalepisodes, cur_step)
        self.writer.add_scalar("test/success_rate", success_num / self.cfg.nevalepisodes, cur_step)

    def test_loop(self, frames: list, titles: list):
        self.cfg.model.eval()
        state, _ = self.env.reset()
        frames.append(self.env.render())
        titles.append("Step 0")
        for step in range(self.cfg.nsteps):
            action = self.cfg.model.get_action(state, deterministic=True)
            transition = self.set_action(state, action)
            frames.append(self.env.render())
            titles.append(f"Step {step+1}")
            if transition.done:
                break
            else:
                state = transition.next_state

    def train(self):
        frames = []
        titles = []
        anim_path = os.path.join(self.cfg.output_dir, f"{self.cfg.basename}-1.mp4")
        try:
            self.cfg.fe.model.eval()
            self.cfg.model.train()
            self.train_loop(frames=frames, titles=titles)
        except KeyboardInterrupt:
            self.cfg.model.save(os.path.join(self.cfg.output_dir, f"{self.cfg.basename}.pth"))
            anim(frames, titles=titles, filename=anim_path, show=False)
            sys.exit(1)
        anim(frames, titles=titles, filename=anim_path, show=False)

    def test(self):
        frames = []
        titles = []
        for _ in range(10):
            self.test_loop(frames=frames, titles=titles)
        anim(
            frames, titles=titles, filename=os.path.join(self.cfg.output_dir, f"{self.cfg.basename}-2.mp4"), show=False
        )
        self.cfg.model.save(os.path.join(os.getcwd(), "model", f"{self.cfg.basename}.pth"))
        self.cfg.model.save(os.path.join(self.cfg.output_dir, f"{self.cfg.basename}.pth"))

    def __call__(self):
        self.train()
        self.test()
        self.close()


class DAExecuter:
    def __init__(self, cfg: DAConfig) -> None:
        self.cfg = cfg
        self.writer = SummaryWriter(log_dir=cfg.output_dir)

        self.sim_imgs = np.load(cfg.sim_data_path)
        self.real_imgs = np.load(cfg.real_data_path)

        np.save(os.path.join(cfg.output_dir, cfg.sim_data_path), self.sim_imgs)
        np.save(os.path.join(cfg.output_dir, cfg.real_data_path), self.real_imgs)

        train_sim, test_sim = th_data.random_split(self.sim_imgs[:, : cfg.fe.img_channel], [0.8, 0.2])
        train_real, test_real = th_data.random_split(self.real_imgs[:, : cfg.fe.img_channel], [0.8, 0.2])
        train_dataset = UnalignedDataset(train_sim, train_real, random=True)
        self.train_dataloader = th_data.DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)
        self.train_data_size = len(train_dataset)
        test_dataset = UnalignedDataset(test_sim, test_real, random=False)
        self.test_dataloader = th_data.DataLoader(dataset=test_dataset, batch_size=cfg.batch_size, shuffle=False)
        self.test_data_size = len(test_dataset)

        model_path1 = os.path.join(MODEL_DIR, cfg.basename + ".pth")
        model_path2 = os.path.join(cfg.output_dir, cfg.basename + ".pth")
        self.es = EarlyStopping(patience=cfg.es_patience, paths=[model_path1, model_path2], trace_func=tqdm.write)

        self.cfg.fe.model.eval()

        self.make_aliases()

    def make_aliases(self):
        self.model = self.cfg.model

    def train(self):
        loss_keys = self.model.loss_names
        for epoch in tqdm(range(1, self.cfg.nepochs + 1)):
            train_loss = {key: 0.0 for key in loss_keys}
            self.model.train()
            for data in self.train_dataloader:
                self.model.set_input(data)
                self.model.optimize_parameters()
                loss = self.model.get_current_losses()
                for key, value in loss.items():
                    train_loss[key] += value * len(self.model.real_A)

            test_loss = {key: 0.0 for key in loss_keys}
            self.model.eval()
            with th.no_grad():
                for data in self.test_dataloader:
                    self.model.set_input(data)
                    self.model.forward()
                    self.model.calcurate_loss_G()
                    self.model.calcurate_loss_D_A()
                    self.model.calcurate_loss_D_B()
                    loss = self.model.get_current_losses()
                    for key, value in loss.items():
                        test_loss[key] += value * len(self.model.real_A)

            for key, value in train_loss.items():
                self.writer.add_scalar(f"train/{key}", value / self.train_data_size, epoch)
            for key, value in test_loss.items():
                self.writer.add_scalar(f"test/{key}", value / self.test_data_size, epoch)

            if check_freq(self.cfg.nepochs, epoch, self.cfg.save_recimg_num):
                for key, value in self.model.get_current_visuals().items():
                    self.writer.add_image(f"rgb/{key}", value[0, :3] * 0.5 + 0.5, epoch)
                    if self.cfg.fe.img_channel > 3:
                        self.writer.add_image(f"depth/{key}", value[0, 3:] * 0.5 + 0.5, epoch)

            if self.es(test_loss["G"], self.model):
                break

    def close(self):
        self.writer.flush()
        self.writer.close()

    def __call__(self):
        try:
            self.train()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()
