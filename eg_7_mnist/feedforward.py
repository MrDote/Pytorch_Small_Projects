import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torchvision as tv
from torchvision import transforms
import matplotlib.pyplot as plt

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_SIZE = 28*28
HIDDEN_SIZE = 10
NUM_CLASSES = 10
EPOCHS = 2
BATCHES = 100
RATE = 0.001

train_dataset = tv.datasets.MNIST(root='./eg_7_mnist/data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = tv.datasets.MNIST(root='./eg_7_mnist/data', train=False, transform=transforms.ToTensor())

# index = torch.arange(10000)
# train_dataset = Subset(train_dataset, index)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCHES, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCHES, shuffle=False)

# #? see first batch 
# eg = iter(train_loader)
# samples, labels = next(eg)
# print(samples.shape, labels.shape)

# #? view whole dataset
# tot = 0
# for i, (images, lables) in enumerate(train_loader):
#     tot += images.shape[0]
# print(tot)

# # #? plot first batch images
# # for i in range(10):
# #     plt.subplot(2, 5, i+1)
# #     plt.imshow(samples[i][0], cmap='gray')
# # plt.show()


class NeuralNet(nn.Module):
    def __init__(self, input, hidden, output):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden, output)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        return out

network = NeuralNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).cuda()


loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=RATE)
total_steps = len(train_loader)

for epoch in range(EPOCHS):
        # images.shape == [10, 1, 28, 28]
    for i, (images, labels) in enumerate(train_loader):
        images = torch.reshape(images, (-1, 28*28)).cuda()
        labels = labels.cuda()

        # forward
        outputs = network(images)
        ls = loss(outputs, labels)

        # backward
        optimizer.zero_grad()
        ls.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {EPOCHS}, step {i+1}/{total_steps}, loss = {ls.item():.4f}')

# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = torch.reshape(images, (-1, 28*28)).cuda()
        labels = labels.cuda()
        outputs = network(images)

        # value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'accuracy: {acc}')