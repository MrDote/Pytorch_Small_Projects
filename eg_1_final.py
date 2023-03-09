import torch
import torch.nn as nn

# training sample
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)

# true values
Y = X.clone()
# f(x) =
Y = Y*2

n_samples, n_features = X.shape
# print(n_samples, n_features)

input_size = n_features
output_size = n_features

class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)

# to be predicted
predict_for = torch.tensor([5], dtype=torch.float32)

rate = 0.1
iters = 100



loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=rate)

print(f'prediction for f({predict_for.item()}) before: {model(predict_for).item():.3f}')

for epoch in range(iters):

    pred = model(X)

    l = loss(Y, pred)

    # backward pass = dl/dw
    l.backward()

    # update weights
    optimizer.step()

    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0 or epoch == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.5f}')

print(f'prediction for f({predict_for.item()}) after: {model(predict_for).item():.3f}')