"""@package RL4CARP
A State Critic is used to estimate the problem complexity

The state critic looks at the log probabilities predicted by the encoder and decoder
and returns the estimate of the complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Encoder


class StateCritic(nn.Module):
    """Estimates the problem complexity.
    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity

    Parameters
    ----------
    static_size: int
        size of the static inputs that will be given to the network
    dynamic_size: int
        size of the dynamic inputs that will be given to the network
    hidden_size: int
        size for constructing the hidden layers
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1)
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):
        """Performs a forward pass on the network

        Parameters
        ----------
        static: torch.Tensor
            static values (x,y coordinates)
        dynamic: torch.Tensor
            current dynamic values (loads, demands)
        """

        # Use the probabilities of visiting each
        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic)

        hidden = torch.cat((static_hidden, dynamic_hidden), 1)

        output = F.relu(self.fc1(hidden))
        output = F.relu(self.fc2(output))
        output = self.fc3(output).sum(dim=2)
        return output