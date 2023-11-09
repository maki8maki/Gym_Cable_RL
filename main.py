import matplotlib.pyplot as plt
import gymnasium as gym
import gym_cable

if __name__ == '__main__':
    gym_cable.register_robotics_envs()
    env = gym.make("MZ04CableGrasp-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    img = env.render()
    plt.imshow(img)
    plt.show()