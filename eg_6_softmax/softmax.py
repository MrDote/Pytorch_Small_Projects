import torch
import torch.nn as nn
import numpy as np

def softmax(list):
    return np.exp(list) / np.sum(np.exp(list), axis=0)

x = np.array([2, 0.2, 0.1], dtype=float)
output = softmax(x)
print(output)

x = torch.from_numpy(x)
output = torch.softmax(x, dim=0)
print(output)