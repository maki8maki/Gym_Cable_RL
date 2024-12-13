import dataclasses
import os
from copy import deepcopy

import dacite
import gymnasium as gym
import hydra
import torch as th
import torch.nn as nn
from absl import logging
from omegaconf import OmegaConf
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

import gym_cable
from agents import utils
from agents.CycleGAN import FeatureExtractionCycleGAN
from callback import MyEvalVallback, VideoRecordCallback
from utils import set_seed
from wrapper import FEWrapper

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def check_device(device: str) -> str:
    if device == "cpu":
        logging.warning("You are using CPU!!")
    if device == "cuda" and not th.cuda.is_available():
        device = "cpu"
        logging.warning("Device changed to CPU!!")
    return device


@dataclasses.dataclass
class FEConfig:
    img_width: int = 64
    img_height: int = 64
    img_channel: int = 4
    hidden_dim: int = 20
    model_name: str = ""
    _model: dataclasses.InitVar[dict] = None
    model: utils.FE = dataclasses.field(default=None)
    _trans: dataclasses.InitVar[dict] = None
    trans: nn.Module = dataclasses.field(default=None, repr=False)

    def convert(self, _cfg: OmegaConf):
        self_copy = deepcopy(self)
        if _cfg._model:
            self_copy.model = hydra.utils.instantiate(_cfg._model)
        if _cfg._trans:
            self_copy.trans = hydra.utils.instantiate(_cfg._trans)
        return self_copy


@dataclasses.dataclass
class TrainFEConfig:
    fe: FEConfig
    basename: str
    _env: dataclasses.InitVar[dict]
    env: gym.Env = dataclasses.field(default=None)
    log_name: dataclasses.InitVar[str] = "grasp_rgbd"
    nsteps: int = dataclasses.field(default=100, repr=False)
    position_random: bool = False
    posture_random: bool = False
    batch_size: int = 128
    nepochs: int = 500
    es_patience: int = 10
    device: str = "cpu"
    seed: dataclasses.InitVar[int] = None
    gathering_data: bool = True
    with_init: bool = True
    data_size: int = 10000
    data_name: str = dataclasses.field(default=None)
    save_recimg_num: int = dataclasses.field(default=10, repr=False)
    output_dir: str = dataclasses.field(default=None)

    def __post_init__(self, _env, log_name, seed):
        if self.fe.model is not None:
            self.fe.model.to(self.device)
        if self.with_init:
            init = "w-init"
        else:
            init = "wo-init"
        if self.position_random:
            position_random = "r"
        else:
            position_random = "s"
        if self.posture_random:
            posture_random = "r"
        else:
            posture_random = "s"
        set_seed(seed)
        self.device = check_device(self.device)
        self.basename += f"_{position_random}{posture_random}_{init}"
        self.data_name = f"{log_name}_{position_random}{posture_random}_{init}_{self.data_size}.npy"
        self.fe.model_name = self.fe.model_name.replace(".pth", f"_{position_random}{posture_random}_{init}.pth")
        self.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    @classmethod
    def convert(cls, _cfg: OmegaConf):
        cfg = dacite.from_dict(data_class=cls, data=OmegaConf.to_container(_cfg))
        cfg.fe = cfg.fe.convert(OmegaConf.create(_cfg.fe))
        cfg.fe.model.to(cfg.device)

        gym_cable.register_robotics_envs()
        cfg.env = gym.make(**_cfg._env)

        return cfg


@dataclasses.dataclass
class SB3Config:
    fe: FEConfig
    basename: str
    _env: dataclasses.InitVar[dict]
    _model: dataclasses.InitVar[dict]
    env: gym.Env = dataclasses.field(default=None)
    nsteps: int = 100
    position_random: bool = False
    posture_random: bool = False
    fe_with_init: dataclasses.InitVar[bool] = True
    device: str = "cpu"
    total_steps: int = 5e5
    seed: int = None
    nevalepisodes: int = 5
    eval_num: int = 1000
    video_num: int = 100
    model: BaseAlgorithm = dataclasses.field(default=None)
    callbacks: CallbackList = dataclasses.field(default=None, repr=False)
    output_dir: str = dataclasses.field(default=None)

    def __post_init__(self, _env, _model, fe_with_init):
        if fe_with_init:
            init = "w-init"
        else:
            init = "wo-init"
        if self.position_random:
            position_random = "r"
        else:
            position_random = "s"
        if self.posture_random:
            posture_random = "r"
        else:
            posture_random = "s"
        set_seed(self.seed)
        self.device = check_device(self.device)
        self.fe.model_name = self.fe.model_name.replace(".pth", f"_{position_random}{posture_random}_{init}.pth")
        self.basename += f"_{position_random}{posture_random}"
        self.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    @classmethod
    def convert(cls, _cfg: OmegaConf):
        cfg = dacite.from_dict(data_class=cls, data=OmegaConf.to_container(_cfg))
        cfg.fe = cfg.fe.convert(OmegaConf.create(_cfg.fe))
        cfg.fe.model.to(device=cfg.device)
        cfg.fe.model.load_state_dict(th.load(os.path.join(MODEL_DIR, cfg.fe.model_name), map_location=cfg.device))
        th.save(cfg.fe.model.state_dict(), os.path.join(cfg.output_dir, cfg.fe.model_name))

        gym_cable.register_robotics_envs()
        env = gym.make(**_cfg._env)
        cfg.env = FEWrapper(env=env, model=cfg.fe.model, trans=cfg.fe.trans)
        cfg.model = hydra.utils.instantiate(
            _cfg._model, env=cfg.env, tensorboard_log=cfg.output_dir, seed=cfg.seed, device=cfg.device
        )
        eval_callback = MyEvalVallback(
            eval_env=Monitor(cfg.env),
            n_eval_episodes=cfg.nevalepisodes,
            best_model_save_filenames=[
                os.path.join(cfg.output_dir, cfg.basename),
                os.path.join("./model", cfg.basename),
            ],
            log_path=cfg.output_dir,
            eval_freq=int(cfg.total_steps / cfg.eval_num),
            verbose=0,
            deterministic=True,
            render=False,
        )
        video_callback = VideoRecordCallback(
            env=cfg.env,
            save_freq=int(cfg.total_steps / cfg.video_num),
            save_filename=os.path.join(cfg.output_dir, f"{cfg.basename}_learning.mp4"),
            deterministic=True,
            verbose=0,
        )
        cfg.callbacks = CallbackList([eval_callback, video_callback])
        return cfg


@dataclasses.dataclass
class DAConfig:
    fe: FEConfig
    basename: str
    _model: dataclasses.InitVar[dict]
    real_data_path: str
    sim_data_path: str
    model: FeatureExtractionCycleGAN = dataclasses.field(default=False)
    position_random: bool = dataclasses.field(default=False, repr=False)
    posture_random: bool = dataclasses.field(default=False, repr=False)
    fe_with_init: dataclasses.InitVar[bool] = dataclasses.field(default=True, repr=False)
    nepochs: int = 100
    es_patience: int = 10
    batch_size: int = 128
    device: str = "cpu"
    seed: dataclasses.InitVar[int] = None
    save_recimg_num: int = dataclasses.field(default=10, repr=False)
    output_dir: str = dataclasses.field(default=None)

    def __post_init__(self, _model, fe_with_init, seed):
        if fe_with_init:
            init = "w-init"
        else:
            init = "wo-init"
        if self.position_random:
            position_random = "r"
        else:
            position_random = "s"
        if self.posture_random:
            posture_random = "r"
        else:
            posture_random = "s"
        set_seed(seed)
        self.device = check_device(self.device)
        self.fe.model_name = self.fe.model_name.replace(".pth", f"_{position_random}{posture_random}_{init}.pth")

        self.real_data_path = os.path.join(DATA_DIR, self.real_data_path)
        self.sim_data_path = os.path.join(DATA_DIR, self.sim_data_path)
        self.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    @classmethod
    def convert(cls, _cfg: OmegaConf):
        cfg = dacite.from_dict(data_class=cls, data=OmegaConf.to_container(_cfg))

        cfg.fe = cfg.fe.convert(OmegaConf.create(_cfg.fe))
        cfg.fe.model.to(cfg.device)
        cfg.fe.model.load_state_dict(th.load(os.path.join(MODEL_DIR, cfg.fe.model_name), map_location=cfg.device))
        th.save(cfg.fe.model.state_dict(), os.path.join(cfg.output_dir, cfg.fe.model_name))

        cfg.model = hydra.utils.instantiate(
            _cfg._model,
            fe=cfg.fe.model,
            input_channel=cfg.fe.img_channel,
            output_channel=cfg.fe.img_channel,
            device=cfg.device,
        )
        cfg.model.setup(cfg.nepochs, cfg.nepochs)
        cfg.model.to(cfg.device)

        return cfg
