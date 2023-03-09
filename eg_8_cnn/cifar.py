import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision as tv
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 1
BATCHES = 10
RATE = 0.005

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCHES, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCHES, shuffle=False)

# eg = iter(train_loader)
# samples, labels = next(eg)
# print(samples.shape, labels.shape)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class ConvNet(nn.Module):
    def __init__(self):
        # input: 32x32
        super(ConvNet, self).__init__()
        # for conv layer output: (input width - filter size + 2*padding)/stride + 1
                                    # 32             5          0           1
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # init: [3, 32, 32]
        x = self.pool(F.relu(self.conv1(x)))
        # before pool: [6, 28, 28]
        # after pool: [6, 14, 14]
        x = self.pool(F.relu(self.conv2(x)))
        # before pool: [16, 10, 10]
        # after pool: [16, 5, 5]
        x = torch.reshape(x, (-1, 16*5*5))
        # [16*5*5]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


model = ConvNet().cuda()

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=RATE)

total_steps = len(train_loader)

for epoch in range(EPOCHS):
    for batch, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()
        # image.shape == [10, 3, 32, 32]
        # labels.shape == [10]

        # forward
        outputs = model(images)
        ls = loss(outputs, labels)

        # backward
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()

        if (batch+1) % 100 == 0:
            print(f'epoch {epoch+1} / {EPOCHS}, step {batch+1}/{total_steps}, loss = {ls.item():.4f}')



# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     n_class_correct = [0 for i in range(10)]
#     n_class_samples = [0 for i in range(10)]
#     for images, labels in test_loader:
#         images = images.cuda()
#         labels = labels.cuda()
#         outputs = model(images)

#         # value, index
#         _, predictions = torch.max(outputs, 1)
#         n_samples += labels.shape[0]
#         n_correct += (predictions == labels).sum().item()

#         for i in range(BATCHES):
#             label = labels[i]
#             prediction = predictions[i]

#             n_class_samples[label] += 1
#             if (label == prediction):
#                 n_class_correct[label] += 1

#     acc = 100.0 * n_correct / n_samples
#     print(f'network accuracy: {acc} %')

#     for clase in range(10):
#         acc = 100.0 * n_class_correct[clase] / n_class_samples[clase]
#         print(f'{classes[clase]} accuracy: {acc} %')