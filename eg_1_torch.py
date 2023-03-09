import torch

# training sample
X = torch.tensor([1,2,3,4], dtype=torch.float32)

# true values
Y = X.clone()
# f(x) =
Y = Y*2
print(Y)

# to be predicted
predict_for = 5

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
rate = 0.01
iters = 40

# model prediction
def forward(x):
    return w * x

# loss = MSE = 1/N*(wx-y)^2
def loss(predicted, real):
    return ((real-predicted)**2).mean()
#  

print(f'prediction for f({predict_for}) before: {forward(predict_for)}')

for epoch in range(iters):
    
    pred = forward(X)

    l = loss(pred, Y)

    # backward pass = dl/dw
    l.backward()

    # update weights
    with torch.no_grad():
        w -= rate * w.grad
    
    w.grad.zero_()

    print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.5f}')

print(f'prediction for f({predict_for}) after: {forward(predict_for)}')