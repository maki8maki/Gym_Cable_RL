import os
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import gym_cable  # noqa: E402

gym_cable.register_robotics_envs()
env = gym.make(
    id="MZ04CableGrasp-v0",
    render_mode="rgb_array",
    max_episode_steps=100,
    position_random=False,
    posture_random=False,
    with_continuous=False,
    cable_width=0.02,
    circuit_width=0.054,
)

obs, _ = env.reset()

keys = ["rgb_image", "depth_image"]
normalized_state = {}
for key in keys:
    state_mean = 0.5 * (env.observation_space[key].high + env.observation_space[key].low)
    state_halfwidth = 0.5 * (env.observation_space[key].high - env.observation_space[key].low)
    normalized_state[key] = ((obs[key].astype(np.float32) - state_mean) / state_halfwidth).astype(np.float32)
rgb: np.ndarray = normalized_state["rgb_image"] * 0.5 + 0.5
depth: np.ndarray = normalized_state["depth_image"] * 0.5 + 0.5
print(rgb.shape, depth.shape)
img = np.concatenate([rgb, depth], axis=-1)
img = img.reshape((1, *img.shape))
print(img.shape)

plt.imshow(depth, vmin=0, vmax=1, cmap="gray")
plt.show()

np.save("data/sim", img)
