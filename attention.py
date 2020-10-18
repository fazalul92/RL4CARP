"""@package RL4CARP
The Attention layer for identifying which input variables are
required to identify the next output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state.

    Parameters
    ----------
    hidden_size: int
        size of the hidden layer
    """

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):
        """ Performs a forward pass on the network

        Parameters
        ---------
        static_hidden: torch.Tensor
            input given to the static layer (x,y coordinates)
        dynamic_hidden: torch.Tensor
            input given to the dynamic layer (loads, demands)
        decoder_hidden: torch.Tensor
            input given to the decoder layer
        """

        batch_size, hidden_size, _ = static_hidden.size()

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)  # (batch, seq_len)
        return attns