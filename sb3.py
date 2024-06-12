import hydra
from omegaconf import OmegaConf

from config import SB3Config
from utils import yes_no_input


@hydra.main(config_path="conf/", config_name="sb3", version_base=None)
def main(_cfg: OmegaConf):
    cfg = SB3Config.convert(_cfg)
    print(f"\n{cfg}\n")

    if not yes_no_input("fe.model_name and basename"):
        exit()

    cfg.model.learn(cfg.total_steps, callback=cfg.callbacks, progress_bar=True)


if __name__ == "__main__":
    main()
