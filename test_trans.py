import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agents.utils import CropTrans  # noqa: E402

trans = CropTrans(64)

path = "data/real_rgbd_quarter.npy"

orig = np.load(path)
print(orig.shape, orig.min(), orig.max())
transed = []

for i in range(len(orig)):
    transed.append(trans(orig[i]))

transed = np.array(transed) * 2 - 1
print(transed.shape, transed.min(), transed.max())

np.save(path, transed)
