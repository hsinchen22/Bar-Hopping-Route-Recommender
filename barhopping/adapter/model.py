import torch.nn as nn

class LinearAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    def forward(self, x):
        return self.linear(x)