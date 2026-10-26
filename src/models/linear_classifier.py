from torch import nn
import torch
import torch.nn.functional as F

class LinearClassifier(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 1)

    def forward(self, x):
        return self.linear(x)

