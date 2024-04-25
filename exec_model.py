import os

import hydra
import matplotlib.pyplot as plt
import torch as th
from omegaconf import OmegaConf

from config import TrainFEConfig
from executer import FEExecuter


@hydra.main(config_path="conf/train_fe", config_name="config", version_base=None)
def main(_cfg: OmegaConf):
    cfg = TrainFEConfig.convert(_cfg)
    print(f"\n{cfg}\n")

    executer = FEExecuter(env_name="MZ04CableGrasp-v0", cfg=cfg)
    del cfg, _cfg

    executer.cfg.fe.model.load_state_dict(
        th.load(os.path.join("model", executer.cfg.fe.model_name), map_location=executer.cfg.device)
    )

    executer.cfg.fe.model.eval()
    state = executer.reset_get_state()
    x = th.tensor(state["image"]).to(executer.cfg.device)
    y = executer.test(x)

    x = x.cpu().squeeze().detach().numpy().transpose(1, 2, 0)
    y = y.cpu().squeeze().detach().numpy().transpose(1, 2, 0)

    plt.axis("off")
    plt.imshow(x[..., :3])
    plt.show()

    plt.axis("off")
    plt.imshow(x[..., 3:], cmap="gray")
    plt.show()

    plt.axis("off")
    plt.imshow(y[..., :3])
    plt.show()

    plt.axis("off")
    plt.imshow(y[..., 3:], cmap="gray")
    plt.show()

    executer.close()


if __name__ == "__main__":
    main()
