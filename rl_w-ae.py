import hydra
from omegaconf import OmegaConf

from utils import yes_no_input
from config import CombConfig
from executer import CombExecuter

@hydra.main(config_path='conf/', config_name='config', version_base=None)
def main(_cfg: OmegaConf):
    cfg = CombConfig.convert(_cfg)
    print(f'\n{cfg}\n')
    
    if not yes_no_input('fe.model_name and basename'):
        exit()
    
    executer = CombExecuter(env_name='MZ04CableGrasp-v0', cfg=cfg)
    del cfg, _cfg

    executer()

if __name__ == '__main__':
    main()
