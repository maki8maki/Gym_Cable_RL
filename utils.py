import random

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False
        if th.cuda.is_available():
            th.cuda.manual_seed(seed)
            th.cuda.manual_seed_all(seed)


def anim(frames, titles=None, filename=None, show=True, interval=50):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=144)
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])
        if titles is not None:
            plt.title(titles[i], fontsize=32)

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=interval)
    if filename is not None:
        anim.save(filename, writer="ffmpeg")
    if show:
        plt.show()


def yes_no_input(check):
    while True:
        print("\n### Please check \033[31m" + check + "\033[0m ###")
        choice = input("Are you sure you want to continue running this code? [y/N]: ").lower()
        if choice in ["y", "ye", "yes"]:
            return True
        elif choice in ["n", "no"]:
            return False


def check_freq(total: int, current: int, num: int):
    return current % (total / num) == 0


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, paths=["checkpoint.pth"], trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.paths = paths
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, val_loss, model: nn.Module):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        for path in self.paths:
            th.save(model.state_dict(), path)
        self.val_loss_min = val_loss
