import torch
import torch.nn as nn

class CEModel(nn.Module):
    def __init__(self):
        super(CEModel, self).__init__()

        self.linear1 = torch.nn.Linear(1024, 512)
        self.linear2 = torch.nn.Linear(512, 2)

        self.activation = torch.nn.ReLU()

    def forward(self, first, second):
        x = torch.cat((first, second), dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x