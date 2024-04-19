from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def plot(log_dir, tagname=None, title=None, xlabel=None, ylabel=None):
    log_files = [path for path in Path(log_dir).glob("**/*") if path.is_file()]

    for log_file in log_files:
        event = EventAccumulator(str(log_file))
        event.Reload()

        tags = event.Tags()["scalars"]

        for tag in tags:
            if tagname is not None and tag != tagname:
                continue
            scalars = event.Scalars(tag)
            step = []
            value = []
            for scalar in scalars:
                step.append(scalar.step)
                value.append(scalar.value)
            x = np.array(step)
            y = np.array(value)
            plt.figure(figsize=(7, 5.6))
            plt.plot(x, y)
            plt.tick_params(labelsize=15)
            if title is not None:
                plt.title(title)
            else:
                plt.title(tag)
            if xlabel is not None:
                plt.xlabel(xlabel, fontsize=18)
            if ylabel is not None:
                plt.ylabel(ylabel, fontsize=18)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    log_dir = "logs/SAC_w-TrainedDCAE/20231227-1913_xyz"
    tagname = "test/reward"
    plot(log_dir, tagname, title="", xlabel="episode", ylabel="rewards")
