import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import gym_cable

from utils import anim

if __name__ == '__main__':
    gym_cable.register_robotics_envs()
    env = gym.make("MZ04CableGrasp-v0", render_mode="rgb_array")

    obs, info = env.reset(seed=42)
    frames = [env.render()]
    for i in range(10):
        action = np.array([0, 0, 0, 0, 0, 0])
        next_obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        # if terminated or truncated:
        #     obs, info = env.reset()
    anim(frames)
    env.close()