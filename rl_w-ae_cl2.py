import torch
from absl import logging
import os
import hydra
from omegaconf import OmegaConf
import dacite
import gymnasium as gym
import gym_cable

from utils import set_seed, anim
from agents.SAC import SAC
from agents.DCAE import DCAE
from config import Config
from executer import CLExecuter

@hydra.main(config_path='conf/', config_name='config', version_base=None)
def main(_cfg: OmegaConf):
    cfg = dacite.from_dict(data_class=Config, data=OmegaConf.to_container(_cfg))
    print(cfg)
    
    seed = 42
    set_seed(seed)
    
    if cfg.device=='cpu':
        logging.warning('You are using CPU!!')
    
    gym_cable.register_robotics_envs()
    env = gym.make("MZ04CableGrasp-v0", render_mode="rgb_array", max_episode_steps=cfg.nsteps, is_random=False)

    fe_config = cfg.get_fe_cfg_dict()
    model = DCAE(**fe_config).to(cfg.device)
    model.load_state_dict(torch.load(f'./model/{cfg.fe.model_name}', map_location=cfg.device))
    model.eval()
    
    rl_config = cfg.get_rl_cfg_dict(env.observation_space['observation'], env.action_space)
    agent = SAC(**rl_config)
    
    cl_scheduler = []
    options = {'diff_ratio': 0.1}
    eps = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
    for ep in eps:
        cl_scheduler.append([ep, options])
        options['diff_ratio'] += 0.1
    
    cl_executer = CLExecuter(env=env, cfg=cfg, fe=model, rl=agent, cl_scheduler=cl_scheduler)

    cl_executer.gathering_data(rl_config['action_space'])

    # train and eval
    frames = []
    titles = []
    cl_executer.train_epsiode_loop(frames=frames, titles=titles)
    anim(frames, titles=titles, filename=f'{cfg.output_dir}/{cfg.basename}-1.mp4', show=False)

    # test
    frames = []
    titles= []
    cl_executer.test_step_loop(frames=frames, titles=titles)
    anim(frames, titles=titles, filename=f'{cfg.output_dir}/{cfg.basename}-2.mp4', show=False)
    cl_executer.rl.save(os.path.join(os.getcwd(), 'model', f'{cfg.basename}.pth'))
    cl_executer.rl.save(os.path.join(cfg.output_dir, f'{cfg.basename}.pth'))

    cl_executer.close()

if __name__ == '__main__':
    main()
