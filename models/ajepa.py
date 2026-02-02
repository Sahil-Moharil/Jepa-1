import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class AJEPA(nn.Module):
    def __init__(self, input_dim=80, embed_dim=384):
        super().__init__()
        self.context_encoder = Encoder(input_dim, embed_dim)
        self.target_encoder = Encoder(input_dim, embed_dim)
        self.predictor = Predictor(embed_dim)

    def forward(self, x_context, x_target):
        z_context = self.context_encoder(x_context)
        z_target = self.target_encoder(x_target).detach()
        z_pred = self.predictor(z_context)
        return z_pred, z_target
