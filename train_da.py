import hydra
from omegaconf import OmegaConf

from config import DAConfig
from executer import DAExecuter
from utils import yes_no_input


@hydra.main(config_path="conf/train_da", config_name="config", version_base=None)
def main(_cfg: OmegaConf):
    cfg = DAConfig.convert(_cfg)
    print(f"\n{cfg}\n")

    if not yes_no_input("fe.model_name, basename and data_name"):
        exit()

    executer = DAExecuter(cfg=cfg)
    del cfg, _cfg

    executer()


if __name__ == "__main__":
    main()
