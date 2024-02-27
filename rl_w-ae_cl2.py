import numpy as np
import hydra
from omegaconf import OmegaConf

from utils import yes_no_input
from config import Config
from executer import CLCombExecuter

@hydra.main(config_path='conf/', config_name='config', version_base=None)
def main(_cfg: OmegaConf):
    cfg = Config.convert(_cfg)
    print(f'\n{cfg}\n')
    
    cl_scheduler = []
    options = {'diff_ratio': 0.1}
    eps = np.array([0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]) * cfg.nepisodes
    for ep in eps:
        cl_scheduler.append([ep, options.copy()])
        options['diff_ratio'] = min(options['diff_ratio']+0.1, 1.0)
    print(cl_scheduler)
    
    if not yes_no_input('fe.model_name and basename'):
        exit()
    
    executer = CLCombExecuter(env_name='MZ04CableGrasp-v0', cfg=cfg, cl_scheduler=cl_scheduler)
    del cfg, _cfg

    executer()

if __name__ == '__main__':
    main()
