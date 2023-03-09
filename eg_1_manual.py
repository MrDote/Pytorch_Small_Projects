import numpy as np

# training sample
X = np.array([1,2,3,4], dtype=np.float32)

# true values
Y = np.copy(X)
# f(x) =
Y = Y*100
print(Y)

# to be predicted
predict_for = 5

w = 0
rate = 0.001
iters = 200

# model prediction
def forward(x):
    if isinstance(x, np.ndarray):
        # return relu(((w * x) - (np.ones(x.size)*2)))
        # return relu(w * x)
        return w * x
    else:
        # return relu(((w * x)))
        return w * x

def relu(x):
    if isinstance(x, np.ndarray):
        return np.maximum(x, 0)
    # If both x and y are numbers, return the maximum of the two
    else:
        return max(x, 0)

# loss = MSE = 1/N*(wx-y)^2
def loss(predicted, real):
    return np.mean((real-predicted)**2)

# gradient = d(MSE)/dw = 1/N*2x(wx-y)
def grad(x, predicted, real):
    return np.mean(np.dot(2*x,(predicted-real)))

print(f'prediction for f({predict_for}) before: {forward(predict_for)}')

for epoch in range(iters):
    
    pred = forward(X)

    l = loss(pred, Y)

    g = grad(X, pred, Y)

    # update weights
    w -= rate * g

    print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.3f}')

print(f'prediction for f({predict_for}) after: {forward(predict_for)}')