"""@package RL4CARP
Sample Encoder Layer to be used by multiple networks.
"""

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')


class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution.

    Parameters
    ---------
    input_size: int
        size of the inputs to the network
    hidden_size: int
        size of the hidden layer
    """

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        """Performs a forward pass on the network

        Parameters
        ---------
        input: torch.Tensor
            the input to be passed through the network
        """
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)