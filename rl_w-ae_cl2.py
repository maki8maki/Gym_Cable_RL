import numpy as np
import os
import hydra
from omegaconf import OmegaConf

from utils import anim, yes_no_input
from config import Config
from executer import CLCombExecuter

@hydra.main(config_path='conf/', config_name='config', version_base=None)
def main(_cfg: OmegaConf):
    cfg = Config.convert(_cfg)
    print(f'\n{cfg}\n')
    
    cl_scheduler = []
    options = {'diff_ratio': 0.1}
    eps = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) * cfg.nepisodes
    for ep in eps:
        cl_scheduler.append([ep, options.copy()])
        options['diff_ratio'] = min(options['diff_ratio']+0.1, 1.0)
    print(cl_scheduler)
    
    if not yes_no_input('fe.model_name and basename'):
        exit()
    
    cl_executer = CLCombExecuter(env_name='MZ04CableGrasp-v0', cfg=cfg, cl_scheduler=cl_scheduler)
    del cfg, _cfg

    cl_executer.gathering_data()

    # train and eval
    frames = []
    titles = []
    cl_executer.train_epsiode_loop(frames=frames, titles=titles)
    anim(frames, titles=titles, filename=f'{cl_executer.cfg.output_dir}/{cl_executer.cfg.basename}-1.mp4', show=False)

    # test
    frames = []
    titles= []
    cl_executer.test_step_loop(frames=frames, titles=titles)
    anim(frames, titles=titles, filename=f'{cl_executer.cfg.output_dir}/{cl_executer.cfg.basename}-2.mp4', show=False)
    cl_executer.cfg.rl.model.save(os.path.join(os.getcwd(), 'model', f'{cl_executer.cfg.basename}.pth'))
    cl_executer.cfg.rl.model.save(os.path.join(cl_executer.cfg.output_dir, f'{cl_executer.cfg.basename}.pth'))

    cl_executer.close()

if __name__ == '__main__':
    main()
