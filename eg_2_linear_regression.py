import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib as mpl
import matplotlib.pyplot as plt

X, Y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X.astype(np.float32))
Y = torch.from_numpy(Y.astype(np.float32))
X = torch.reshape(X, (X.shape[0], 1))
Y = torch.reshape(Y, (Y.shape[0], 1))

samples, features = X.shape

input_size = features
output_size = 1

model = nn.Linear(input_size, output_size)

rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=rate)

epochs = 100

for epoch in range(epochs):
    Y_pred = model(X)
    loss = criterion(Y_pred, Y)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

    if (epoch + 1) % 20 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

predicted = model(X).detach().numpy()

plt.plot(X, Y, 'ro')
plt.plot(X, predicted, 'b')
plt.show()