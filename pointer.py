"""@package RL4CARP
The Pointer network for specifying the nodes and the order in which they have to be visited.
"""

import torch
import torch.nn as nn

from attention import Attention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings.

    Parameters
    ----------
    hidden_size: int
        number of nodes in the hidden layer
    num_layers: int, optional
        number of total layers
    dropout: float, optional
        percentage of values to be dropped between layers

    Methods
    -------
    forward(static_hidden, dynamic_hidden, decoder_hidden, last_hh)
        performs a forward pass on the network.
    """

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.encoder_attn = Attention(hidden_size)

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):
        """ Performs a forward pass on the network

        Parameters
        ---------
        static_hidden: torch.Tensor
            input given to the static layer (x,y coordinates)
        dynamic_hidden: torch.Tensor
            input given to the dynamic layer (loads, demands)
        decoder_hidden: torch.Tensor
            input given to the decoder layer
        last_hh: torch.Tensor
            output from the last hidden state
        """
        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)
        rnn_out = rnn_out.squeeze(1)

        # Always apply dropout on the RNN output
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh) 

        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # (B, 1, num_feats)

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden)
        energy = torch.cat((static_hidden, context), dim=1)  # (B, num_feats, seq_len)

        v = self.v.expand(static_hidden.size(0), -1, -1)
        W = self.W.expand(static_hidden.size(0), -1, -1)

        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)

        return probs, last_hh