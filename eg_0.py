import torch
torch.manual_seed(0)
import numpy as np

x = np.array([1,2,3,4], dtype=np.float32)
y = np.array([4,3,2,1], dtype=np.float32)

print(x*y)
print(np.dot(x, y))