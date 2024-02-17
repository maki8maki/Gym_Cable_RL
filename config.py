import dataclasses
import torch
import torch.nn as nn
from absl import logging
from copy import deepcopy
import hydra
from omegaconf import OmegaConf
import dacite

from agents import buffer
from agents import utils
from agents.DCAE import DCAE
from agents.SAC import SAC
from utils import set_seed

@dataclasses.dataclass
class FEConfig:
    img_width: int = 108
    img_height: int = 72
    img_channel : int = 4
    hidden_dim: int = 20
    model_name: str = ''
    _model: dataclasses.InitVar[dict] = None
    model: nn.Module = dataclasses.field(default=None)
    _trans: dataclasses.InitVar[dict] = None
    trans: nn.Module = dataclasses.field(default=None, repr=False)
    
    def __post_init__(self, _model, _trans):
        if _model is None:
            self.model = DCAE(img_height=self.img_height, img_width=self.img_width, img_channel=self.img_channel, hidden_dim=self.hidden_dim)
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
class Config:
    fe: FEConfig
    rl: RLConfig
    basename: str
    nsteps: int = dataclasses.field(default=100, repr=False)
    memory_size: int = 10000
    nepisodes: int = 5000
    nevalepisodes: int = dataclasses.field(default=5, repr=False)
    update_every: int = 50
    save_anim_num: int = dataclasses.field(default=10, repr=False)
    eval_num: int = dataclasses.field(default=100, repr=False)
    device: str = 'cpu'
    seed: dataclasses.InitVar[int] = None
    replay_buffer: buffer.Buffer = dataclasses.field(default=buffer.ReplayBuffer(memory_size=memory_size), repr=False)
    gathering_data: bool = False
    buffer_name: str = dataclasses.field(init=False)
    output_dir: str = './logs'
    
    def __post_init__(self, seed):
        if seed is not None:
            set_seed(seed)
        if self.device == 'cpu':
            logging.warning('You are using CPU!!')
        if self.device=='cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            logging.warning('Device changed to CPU!!')
        if self.fe.model is not None:
            self.fe.model.to(self.device)
        if self.rl.model is not None:
            self.rl.model.to(self.device)
        self.buffer_name = f'buffer_o-{self.rl.obs_dim}_a-{self.rl.act_dim}_w-hs_{self.memory_size}.pcl'
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
        cfg.basename = _cfg.basename
        cfg.replay_buffer = hydra.utils.instantiate(_cfg._replay_buffer)
        return cfg
