import numpy as np
import os
import hydra
from omegaconf import OmegaConf

from utils import anim, yes_no_input
from config import Config
from executer import CombExecuter

@hydra.main(config_path='conf/', config_name='config', version_base=None)
def main(_cfg: OmegaConf):
    cfg = Config.convert(_cfg)
    print(f'\n{cfg}\n')
    
    if not yes_no_input('fe.model_name and basename'):
        exit()
    
    executer = CombExecuter(env_name='MZ04CableGrasp-v0', cfg=cfg)
    del cfg, _cfg

    executer.gathering_data()

    # train and eval
    frames = []
    titles = []
    executer.train_epsiode_loop(frames=frames, titles=titles)
    anim(frames, titles=titles, filename=f'{executer.cfg.output_dir}/{executer.cfg.basename}-1.mp4', show=False)

    # test
    frames = []
    titles= []
    executer.test_step_loop(frames=frames, titles=titles)
    anim(frames, titles=titles, filename=f'{executer.cfg.output_dir}/{executer.cfg.basename}-2.mp4', show=False)
    executer.cfg.rl.model.save(os.path.join(os.getcwd(), 'model', f'{executer.cfg.basename}.pth'))
    executer.cfg.rl.model.save(os.path.join(executer.cfg.output_dir, f'{executer.cfg.basename}.pth'))

    executer.close()

if __name__ == '__main__':
    main()
