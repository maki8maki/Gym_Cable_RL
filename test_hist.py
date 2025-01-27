import matplotlib.pyplot as plt
import numpy as np

data = np.load("data/error.npy")
data = data * -1 + 0.002
data *= 1000

min = data.min()
max = data.max()
step = 1

print(data.mean())
print(data.std())

plt.hist(data, bins=np.arange(min, max + step, step=step))
plt.tick_params(labelsize=15)
plt.xlabel("Distance [mm]", fontsize=18)
plt.ylabel("Frequency", fontsize=18)
plt.tight_layout()
plt.show()
# plt.savefig("logs/dis_hist.png")
