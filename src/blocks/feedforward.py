# Author: Mohammed Rahman
# Student ID: 10971320
# University of Manchester — BSc Computer Science Final Year Project, 2026
#

import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_factor=4, dropout=0.0):
        super().__init__()

        # Hidden layer size is scaled relative to the input dimension
        hidden_dim = dim * hidden_factor

        # Two-layer feedforward network with GELU activation and dropout
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pass input through the feedforward network
        return self.net(x)