import torch
import torch.nn as nn

class TabularNN(nn.Module):
    def __init__(self,num_features):
        super(TabularNN, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.stack(x)
