import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from google.protobuf.json_format import MessageToDict
from tensorboard.backend.event_processing.event_accumulator import (
    DEFAULT_SIZE_GUIDANCE,
    TENSORS,
    EventAccumulator,
    TensorEvent,
)

import gym_cable
from utils import anim

gym_cable.register_robotics_envs()


def playback(log_dir, length=0):
    log_files = [path for path in Path(log_dir).glob("**/events*") if path.is_file()]

    size_guidance = DEFAULT_SIZE_GUIDANCE
    size_guidance.update([(TENSORS, length)])

    id = 0

    for log_file in log_files:
        event = EventAccumulator(str(log_file), size_guidance=size_guidance)
        event.Reload()

        tes: list[TensorEvent] = event.Tensors("action")

        env = gym.make("MZ04CableGrasp-v0", render_mode="rgb_array", position_random=False, posture_random=False)
        env.reset()

        imgs = [env.render()]
        for te in tes:
            tensor_proto = te.tensor_proto
            action = MessageToDict(tensor_proto)["floatVal"]
            env.step(action)
            img = env.render()
            imgs.append(img)

        imgs = np.array(imgs)
        anim(imgs, filename=os.path.join(log_dir, f"playback{id}.mp4"), interval=200)

        id += 1


if __name__ == "__main__":
    log_dir = "logs/real/SB3_SAC_trainedVAE/20240705-2028/"
    playback(log_dir)
