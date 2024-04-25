import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def plot(log_dir, tagnames=None, title=None, xlabel=None, ylabel=None):
    log_files = [path for path in Path(log_dir).glob("**/*") if path.is_file()]

    for log_file in log_files:
        event = EventAccumulator(str(log_file))
        event.Reload()

        tags = event.Tags()["scalars"]

        for tag in tags:
            if tagnames is not None and tag in tagnames:
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


def show_image(log_dir, tagnames=None, is_save=False):
    log_files = [path for path in Path(log_dir).glob("**/*") if path.is_file()]

    for log_file in log_files:
        event = EventAccumulator(str(log_file))
        event.Reload()

        tags = event.Tags()["images"]

        for tag in tags:
            if tagnames is not None and tag not in tagnames:
                continue
            imgs = event.Images(tag)
            img = tf.image.decode_image(imgs[-1].encoded_image_string).numpy()
            print(tag)
            plt.imshow(img)
            plt.axis("off")

            if is_save:
                plt.savefig(os.path.join("logs", tag.replace("/", "_") + ".pdf"))

            plt.show()


if __name__ == "__main__":
    log_dir = "logs/DCAE/20240424-1908"
    tagnames = ["depth/200_reconstructed", "rgb/200_reconstructed", "depth/200_original", "rgb/200_original"]
    show_image(log_dir, tagnames, is_save=True)
