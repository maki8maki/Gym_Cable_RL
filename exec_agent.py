import torch
import hydra
from omegaconf import OmegaConf

from utils import anim
from config import CombConfig
from executer import CombExecuter

@hydra.main(config_path='conf/', config_name='config', version_base=None)
def main(_cfg: OmegaConf):
    cfg = CombConfig.convert(_cfg)
    print(f'\n{cfg}\n')
    
    executer = CombExecuter(env_name='MZ04CableGrasp-v0', cfg=cfg)
    executer.cfg.rl.model.load_state_dict(torch.load('logs/SAC_trainedDCAE_o-3_a-3/20240226-2249/SAC_trainedDCAE_o-3_a-3.pth', map_location=executer.cfg.device))
    
    frames = []
    titles= []
    executer.test_step_loop(frames=frames, titles=titles)
    anim(frames, titles=titles, filename=f'{executer.cfg.output_dir}/{executer.cfg.basename}-2.mp4', show=False)
    
    executer.close()

if __name__ == '__main__':
    main()
