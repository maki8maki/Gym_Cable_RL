import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

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
            plt.plot(x, y)
            if title is not None:
                plt.title(title)
            else:
                plt.title(tag)
            if xlabel is not None:
                plt.xlabel(xlabel)
            if ylabel is not None:
                plt.ylabel(ylabel)
            plt.show()

if __name__ == '__main__':
    log_dir = 'logs/SAC/20231220-2054_only-rl_x-action'
    tagname = 'train/reward'
    plot(log_dir, tagname, 'Episode Rewards', 'episode', 'rewards')
            
