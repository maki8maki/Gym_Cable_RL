import matplotlib.pyplot as plt
import numpy as np

name = "eval_mean_reward"
d = {
    "sac": "SAC",
    "ppo": "PPO",
    "td3": "TD3",
}
# d = {
#     "sac": "Training from scratch (cable width: 30mm)",
#     "sac_20": "Using pre-trained model (cable width: 20mm)",
# }
num = 5

plt.figure(figsize=(7, 5.6))
tick = np.arange(0, 3) * 1e5
label = [str(int(t / 1e3)) + "K" for t in tick]
plt.xticks(tick, label)

for k, v in d.items():
    data = []
    for i in range(num):
        data.append(np.load(f"logs/{name}_{k}_{i}.npy"))
    data = np.array(data)
    x = np.arange(data.shape[1]) * 500
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    plt.fill_between(x, mean - std, mean + std, alpha=0.6)
    plt.plot(x, mean, label=v)

plt.tick_params(labelsize=15)
plt.xlabel("Steps", fontsize=18)
plt.ylabel("Episode Rewards", fontsize=18)
plt.ylim(-2.5, 0)
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig("logs/reward.pdf")
