import dataclasses
from typing import Dict, Tuple
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
import hydra

from agents import utils

@dataclasses.dataclass
class FEConfig:
    model_name: str
    lr: float = 1e-3
    img_width: int = 108
    img_height: int = 72
    img_dim : dataclasses.InitVar[int] = 4
    image_size: Tuple[int, int, int] = dataclasses.field(init=False)
    hidden_dim: int = 20
    _trans: dataclasses.InitVar[dict] = None
    trans: nn.Module = dataclasses.field(init=False)
    _net_activation: dataclasses.InitVar[dict] = None
    net_activation: nn.Module = dataclasses.field(init=False)
    _loss_func: dataclasses.InitVar[dict] = None
    loss_func: nn.Module = dataclasses.field(init=False)
    
    def __post_init__(self, img_dim, _trans, _net_activation, _loss_func):
        self.image_size = (self.img_height, self.img_width, img_dim)
        if _trans is not None:
            self.trans = hydra.utils.instantiate(_trans, img_width=self.img_width, img_height=self.img_height)
        else:
            self.trans = utils.MyTrans(img_width=self.img_width, img_height=self.img_height)
        if _net_activation is not None:
            self.net_activation = hydra.utils.instantiate(_net_activation)
        else:
            self.net_activation = nn.GELU()
        if _loss_func is not None:
            self.loss_func = hydra.utils.instantiate(_loss_func, channel=img_dim)
        else:
            self.loss_func = utils.SSIMLoss(channel=img_dim)
    
    def get_cfg_dict(self):
        # 辞書を作成し、不要なデータの削除を行う
        cfg_dict = dataclasses.asdict(self)
        del_list = ['img_width', 'img_height', 'trans', 'model_name']
        for key in del_list:
            del cfg_dict[key] 
        return cfg_dict

@dataclasses.dataclass
class RLConfig:
    gamma: float = 0.7
    batch_size: int = 50
    lr: float = 1e-3
    obs_dim: int = 6
    act_dim: int = 6
    
    def get_cfg_dict(self):
        # 辞書を作成し、不要なデータの削除を行う
        cfg_dict = dataclasses.asdict(self)
        del_list = ['obs_dim', 'act_dim']
        for key in del_list:
            del cfg_dict[key]
        return cfg_dict

@dataclasses.dataclass
class Config:
    fe: FEConfig
    rl: RLConfig
    basename: str
    nsteps: int = 100
    memory_size: int = 10000
    nepisodes: int = 5000
    nevalepisodes: int = 5
    update_every: int = 50
    save_anim_num: int = 10
    eval_num: int = 100
    device: str = 'cpu'
    _replay_buffer: dataclasses.InitVar[dict] = None
    replay_buffer: utils.Buffer = dataclasses.field(init=False)
    gathering_data: bool = False
    data_name: str = dataclasses.field(init=False)
    output_dir: str = dataclasses.field(init=False)
    
    def __post_init__(self, _replay_buffer):
        if self.device=='cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
        if _replay_buffer is not None:
            self.replay_buffer = hydra.utils.instantiate(_replay_buffer, memory_size=self.memory_size)
        else:
            self.replay_buffer = utils.ReplayBuffer(memory_size=self.memory_size)
        self.basename += f'_o-{self.rl.obs_dim}_a-{self.rl.act_dim}'
        self.data_name = f'buffer_o-{self.rl.obs_dim}_a-{self.rl.act_dim}_w-hs_{self.memory_size}.pcl'
        self.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    def get_fe_cfg_dict(self):
        cfg_dict = self.fe.get_cfg_dict() 
        return cfg_dict
    
    def get_rl_cfg_dict(self, obs_space: spaces.Box=None, act_space: spaces.Box=None):
        cfg_dict = self.rl.get_cfg_dict()

        hidden_low = np.full(self.fe.hidden_dim, -1.0)
        hidden_high = np.full(self.fe.hidden_dim, 1.0)
        if obs_space is None:
            obs_space_low = np.concatenate([hidden_low, np.full(self.rl.obs_dim, -1.0)])
            obs_space_high = np.concatenate([hidden_low, np.full(self.rl.obs_dim, 1.0)])
        else:
            obs_space_low = np.concatenate([hidden_low, obs_space.low[:self.rl.obs_dim]])
            obs_space_high = np.concatenate([hidden_high, obs_space.high[:self.rl.obs_dim]])
        observation_space = spaces.Box(low=obs_space_low, high=obs_space_high, dtype=np.float64)
        if act_space is None:
            action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.rl.act_dim,), dtype=np.float64)
        else:
            action_space = spaces.Box(low=act_space.low[:self.rl.act_dim], high=act_space.high[:self.rl.act_dim], dtype=np.float64)
        
        # データの追加
        cfg_dict['observation_space'] = observation_space
        cfg_dict['action_space'] = action_space

        return cfg_dict
    
    def get_cfg_dict(self, fe_config: Dict[str, any]=None, rl_config: Dict[str, any]=None):
        cfg_dict = dataclasses.asdict(self)
        del_list = ['fe', 'rl', 'device', 'replay_buffer']
        for key in del_list:
            del cfg_dict[key]
        if fe_config is None:
            fe_config = self.fe.get_cfg_dict()
        for key, val in fe_config.items():
            if not isinstance(val, (int, float, str, bool, torch.Tensor)):
                continue
            cfg_dict['fe_'+key] = val
        if rl_config is None:
            rl_config = self.rl.get_cfg_dict()
        for key, val in rl_config.items():
            if not isinstance(val, (int, float, str, bool, torch.Tensor)):
                continue
            cfg_dict['rl_'+key] = val
        return cfg_dict
