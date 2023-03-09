import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

class WineDataset(Dataset):

    # data loading
    def __init__(self, transform=None):
        data = np.loadtxt('eg_5_transforms/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        print(data.shape)
        self.samples = data.shape[0]

        self.x = data[:, 1:]
        self.y = data[:, [0]]

        self.transform = transform

    # have to overwrite
    # allows for indexing
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    # optional to overwrite
    # len(dataset)
    def __len__(self):
        return self.samples

# custom class converts sample from np array to tensor
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor
        
    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


dataset = WineDataset(transform=ToTensor())
first = dataset[0]
features, labels = first
print(features)

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])

dataset = WineDataset(transform=composed)
first = dataset[0]
features, labels = first
print(features)