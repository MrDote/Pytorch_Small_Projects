# linear (relu) -> linear (no softmax)

num_classes = 5

import torch
import torch.nn as nn

class MultiClass(nn.Module):

    def __init__(self, number_inputs, number_hidden, number_output):
        super(MultiClass, self).__init__()
        self.linear1 = nn.Linear(number_inputs, number_hidden)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(number_hidden, number_output)
    
    def forward(self, input):
        x = self.linear1(input)
        x = torch.relu(x)
        y_pred = self.linear2(x)

        return y_pred

model = MultiClass(number_inputs=28*28, number_hidden=5, number_output=num_classes)
# binary cross entropy loss
loss = nn.CrossEntropyLoss()