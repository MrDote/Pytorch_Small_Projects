# linear (relu) -> linear (sigmoid)

import torch
import torch.nn as nn

class BinaryClass(nn.Module):

    def __init__(self, number_inputs, number_hidden):
        super(BinaryClass, self).__init__()
        self.linear1 = nn.Linear(number_inputs, number_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(number_hidden, 1)
    
    def forward(self, input):
        x = self.linear1(input)
        x = torch.relu(x)
        x = self.linear2(x)
        y_pred = torch.sigmoid(x)
        return y_pred

model = BinaryClass(number_inputs=28*28, number_hidden=5)
# binary cross entropy loss
loss = nn.BCELoss()