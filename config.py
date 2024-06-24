import dataclasses
from copy import deepcopy

import dacite
import gymnasium as gym
import hydra
import torch
import torch.nn as nn
from absl import logging
from omegaconf import OmegaConf
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

import gym_cable
from agents import buffer, utils
from agents.DCAE import DCAE
from agents.SAC import SAC
from utils import set_seed
from wrapper import FEWrapper


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

    def __post_init__(self, _model, _trans):
        if _model is None:
            self.model = DCAE(
                img_height=self.img_height,
                img_width=self.img_width,
                img_channel=self.img_channel,
                hidden_dim=self.hidden_dim,
            )
        if _trans is None:
            self.trans = utils.MyTrans(img_width=self.img_width, img_height=self.img_height)

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
    save_recimg_num: int = 10
    output_dir: str = dataclasses.field(default=None)

    def __post_init__(self, log_name, seed):
        if seed is not None:
            set_seed(seed)
        if self.device == "cpu":
            logging.warning("You are using CPU!!")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            logging.warning("Device changed to CPU!!")
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
        self.basename += f"_{position_random}{posture_random}_{init}"
        self.data_name = f"{log_name}_{position_random}{posture_random}_{init}_{self.data_size}.npy"
        self.fe.model_name = self.fe.model_name.replace(".pth", f"_{position_random}{posture_random}_{init}.pth")
        self.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    @classmethod
    def convert(cls, _cfg: OmegaConf):
        cfg = dacite.from_dict(data_class=cls, data=OmegaConf.to_container(_cfg))
        cfg.fe = cfg.fe.convert(OmegaConf.create(_cfg.fe))
        cfg.fe.model.to(cfg.device)
        return cfg


@dataclasses.dataclass
class RLConfig:
    obs_dim: int = 6
    act_dim: int = 6
    _model: dataclasses.InitVar[dict] = None
    model: utils.RL = dataclasses.field(default=None)

    def __post_init__(self, _model):
        if _model is None:
            self.model = SAC(obs_dim=self.obs_dim, act_dim=self.act_dim)

    def convert(self, _cfg: OmegaConf):
        self_copy = deepcopy(self)
        if _cfg._model:
            self_copy.model = hydra.utils.instantiate(_cfg._model)
        return self_copy


@dataclasses.dataclass
class CombConfig:
    fe: FEConfig
    rl: RLConfig
    basename: str
    nsteps: int = dataclasses.field(default=100, repr=False)
    position_random: bool = False
    posture_random: bool = False
    memory_size: int = 10000
    start_steps: int = 10000
    total_steps: int = 5e5
    nevalepisodes: int = dataclasses.field(default=5, repr=False)
    update_after: int = 1000
    update_every: int = 50
    batch_size: int = 50
    save_anim_num: int = dataclasses.field(default=10, repr=False)
    eval_num: int = dataclasses.field(default=100, repr=False)
    device: str = "cpu"
    seed: dataclasses.InitVar[int] = None
    replay_buffer: buffer.Buffer = buffer.ReplayBuffer(memory_size=memory_size)
    fe_with_init: dataclasses.InitVar[bool] = True
    output_dir: str = dataclasses.field(default=None)

    def __post_init__(self, seed, fe_with_init):
        if seed is not None:
            set_seed(seed)
        if self.device == "cpu":
            logging.warning("You are using CPU!!")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            logging.warning("Device changed to CPU!!")
        if self.fe.model is not None:
            self.fe.model.to(self.device)
        if self.rl.model is not None:
            self.rl.model.to(self.device)
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
        self.fe.model_name = self.fe.model_name.replace(".pth", f"_{position_random}{posture_random}_{init}.pth")
        self.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    @classmethod
    def convert(cls, _cfg: OmegaConf):
        cfg = dacite.from_dict(data_class=cls, data=OmegaConf.to_container(_cfg))
        cfg.fe = cfg.fe.convert(OmegaConf.create(_cfg.fe))
        if _cfg.rl._model:
            _cfg.rl._model.obs_dim = cfg.rl.obs_dim + cfg.fe.hidden_dim
        cfg.rl = cfg.rl.convert(OmegaConf.create(_cfg.rl))
        cfg.fe.model.to(cfg.device)
        cfg.rl.model.to(cfg.device)
        cfg.basename = _cfg.basename + ("_r" if cfg.position_random else "_s") + ("r" if cfg.posture_random else "s")
        cfg.replay_buffer = hydra.utils.instantiate(_cfg._replay_buffer)
        return cfg


@dataclasses.dataclass
class SB3Config:
    fe: FEConfig
    basename: str
    _env: dataclasses.InitVar[dict]
    _model: dataclasses.InitVar[dict]
    nsteps: int = dataclasses.field(default=100, repr=False)
    position_random: bool = False
    posture_random: bool = False
    fe_with_init: dataclasses.InitVar[bool] = True
    device: str = "cpu"
    total_steps: int = 5e5
    seed: int = None
    # save_anim_num: int = dataclasses.field(default=10, repr=False)
    nevalepisodes: int = 5
    eval_num: int = 1000
    model: BaseAlgorithm = dataclasses.field(default=None)
    callbacks: CallbackList = dataclasses.field(default=None, repr=False)
    output_dir: str = dataclasses.field(default=None)

    def __post_init__(self, fe_with_init, _env, _model):
        if self.seed is not None:
            set_seed(self.seed)
        if self.device == "cpu":
            logging.warning("You are using CPU!!")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            logging.warning("Device changed to CPU!!")
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
        self.fe.model_name = self.fe.model_name.replace(".pth", f"_{position_random}{posture_random}_{init}.pth")
        self.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    @classmethod
    def convert(cls, _cfg: OmegaConf):
        cfg = dacite.from_dict(data_class=cls, data=OmegaConf.to_container(_cfg))
        cfg.fe = cfg.fe.convert(OmegaConf.create(_cfg.fe))
        cfg.fe.model.to(device=cfg.device)

        gym_cable.register_robotics_envs()
        env = gym.make(
            max_episode_steps=cfg.nsteps,
            position_random=cfg.position_random,
            posture_random=cfg.posture_random,
            **_cfg._env,
        )
        env = FEWrapper(env=env, model=cfg.fe.model, trans=cfg.fe.trans)
        cfg.model = hydra.utils.instantiate(
            _cfg._model, env=env, tensorboard_log=cfg.output_dir, seed=cfg.seed, device=cfg.device
        )
        eval_callback = EvalCallback(
            eval_env=Monitor(env),
            n_eval_episodes=cfg.nevalepisodes,
            best_model_save_path=cfg.output_dir,
            log_path=cfg.output_dir,
            eval_freq=int(cfg.total_steps / cfg.eval_num),
            verbose=0,
            deterministic=True,
            render=False,
        )
        cfg.callbacks = CallbackList([eval_callback])
        return cfg
