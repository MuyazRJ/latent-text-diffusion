import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_factor=4, dropout=0.0):
        super().__init__()
        hidden_dim = dim * hidden_factor

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
