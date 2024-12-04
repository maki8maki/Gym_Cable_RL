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


def tensorevent2value(te: TensorEvent):
    proto = te.tensor_proto
    return MessageToDict(proto)["floatVal"]


def playback(log_dir, length=0):
    np.set_printoptions(precision=5, suppress=True)

    log_files = [path for path in Path(log_dir).glob("**/events*") if path.is_file()]

    size_guidance = DEFAULT_SIZE_GUIDANCE
    size_guidance.update([(TENSORS, length)])

    id = 0

    for log_file in log_files:
        event = EventAccumulator(str(log_file), size_guidance=size_guidance)
        event.Reload()

        ac_tes: list[TensorEvent] = event.Tensors("action")
        pos_tes: list[TensorEvent] = event.Tensors("position")

        env = gym.make(
            "MZ04CableGrasp-v0",
            render_mode="rgb_array",
            position_random=False,
            posture_random=False,
            with_continuous=True,
        )
        obs, _ = env.reset()
        next_obs = obs

        imgs = [env.render()]
        for ac_te, pos_te in zip(ac_tes, pos_tes):
            action = np.array(tensorevent2value(ac_te))
            position = np.array(tensorevent2value(pos_te))
            print("real  :", position)
            print("sim   :", next_obs["observation"])
            next_obs, _, _, _, _ = env.step(action)
            img = env.render()
            imgs.append(img)

        imgs = np.array(imgs)
        anim(imgs, filename=os.path.join(log_dir, f"playback{id}.mp4"), interval=200)

        id += 1


if __name__ == "__main__":
    log_dir = "logs/real/SB3_DA_SAC_trainedVAE/20241114-1948"
    playback(log_dir)
