import torch
import torch.nn as nn
import numpy as np

# NUMPY
def cross_entropy(predicted, actual):
    loss = -np.sum(actual * np.log(predicted))
    return loss
    # normalize
    # return loss / float(predicted.shape[0])

# hot encoded Y => 1 true class
Y = np.array([1,0,0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1,0.3,0.6])

l1 = cross_entropy(Y_pred_good, Y)
l2 = cross_entropy(Y_pred_bad, Y)
# print(l1)
# print(l2)

# TENSOR
def cross_entropy_tensor(predicted, actual):
    loss = -torch.sum(actual * torch.log(predicted))
    return loss
    # normalize
    # return loss / float(predicted.shape[0])

Y = torch.tensor([1,0,0])
Y_pred_good = torch.tensor([0.7, 0.2, 0.1])
Y_pred_bad = torch.tensor([0.1,0.3,0.6])

l1 = cross_entropy_tensor(Y_pred_good, Y)
l2 = cross_entropy_tensor(Y_pred_bad, Y)
# print(l1)
# print(l2)


# TORCH
# nn.CrossEntropyLoss applies softmax & nllloss

loss = nn.CrossEntropyLoss()

# position of correct class
Y = torch.tensor([0])

# shape: nsamples x nclasses = 1x3
Y_pred_good = torch.tensor([[2, 1, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2, 0.3]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1)
print(l2)

_, pred1 = torch.max(Y_pred_good, 1)
_, pred2 = torch.max(Y_pred_bad, 1)

print(pred1)
print(pred2)

# For multiple samples
Y = torch.tensor([2, 0, 1])

Y_pred_good = torch.tensor([[0.7, 1, 2.1], [2, 1, 0.1], [2, 3, 0.1]])
Y_pred_bad = torch.tensor([[1, 2.3, 1.4], [1.2, 2.0, 0.3], [1, 0.4, 0.1]])
l = loss(Y_pred_good, Y)
l1 = loss(Y_pred_bad, Y)
print(l)
print(l1)