import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import (
    DEFAULT_SIZE_GUIDANCE,
    IMAGES,
    EventAccumulator,
    ImageEvent,
    ScalarEvent,
)


def plot(log_dir, tagnames=None, title=None, xlabel=None, ylabel=None, filename=None):
    log_files = [path for path in Path(log_dir).glob("**/*") if path.is_file()]

    for log_file in log_files:
        event = EventAccumulator(str(log_file))
        event.Reload()

        tags = event.Tags()["scalars"]

        for tag in tags:
            if tagnames is not None and tag not in tagnames:
                continue
            scalars: list[ScalarEvent] = event.Scalars(tag)
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
            if xlabel is not None:
                plt.xlabel(xlabel, fontsize=18)
            if ylabel is not None:
                plt.ylabel(ylabel, fontsize=18)
            plt.tight_layout()
            if filename is not None:
                plt.savefig(filename)
            else:
                plt.show()


def save(log_dir, tagnames=None):
    log_files = [path for path in Path(log_dir).glob("**/*") if path.is_file()]

    for log_file in log_files:
        event = EventAccumulator(str(log_file))
        event.Reload()

        tags = event.Tags()["scalars"]

        for tag in tags:
            if tagnames is not None and tag not in tagnames:
                continue
            scalars: list[ScalarEvent] = event.Scalars(tag)
            value = []
            for scalar in scalars:
                value.append(scalar.value)
            np.save(os.path.join("logs", tag.replace("/", "_")), value)


def show_image(log_dir, tagnames=None, is_save=False, log_length=0, index=-1):
    log_files = [path for path in Path(log_dir).glob("**/*") if path.is_file()]

    size_guidance = DEFAULT_SIZE_GUIDANCE
    size_guidance.update([(IMAGES, log_length)])

    for log_file in log_files:
        event = EventAccumulator(str(log_file), size_guidance=size_guidance)
        event.Reload()

        tags = event.Tags()["images"]

        for tag in tags:
            if tagnames is not None and tag not in tagnames:
                continue
            imgs: list[ImageEvent] = event.Images(tag)
            img = tf.image.decode_image(imgs[index].encoded_image_string).numpy()
            fig, _ = plt.subplots(figsize=(img.shape[1] / 10, img.shape[0] / 10))
            plt.imshow(img, vmin=0, vmax=255)
            plt.axis("off")
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            if is_save:
                plt.savefig(os.path.join("logs", tag.replace("/", "_") + f"_{index}.pdf"))

            plt.show()


if __name__ == "__main__":
    log_dir = "logs/VAE/20240717-1812"
    tagnames = ["depth/reconstructed", "rgb/reconstructed", "depth/original", "rgb/original"]
    show_image(log_dir, tagnames, is_save=True)
    # plot(log_dir, ["eval/mean_reward"], xlabel="Steps", ylabel="Episode Rewards", filename="logs/reward.pdf")
