from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, input_length=None):
        super(VanillaRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_length = input_length  

        # input to hidden
        self.Whx = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        # hidden to hidden
        self.Whh = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        # hidden to output
        self.Wph = nn.Linear(self.hidden_dim, output_dim, bias=True)

    def forward(self, x):
        
        if self.input_length and x.size(1) != self.input_length:
            raise ValueError(f"Expected sequence length of {self.input_length}, but got {x.size(1)}")

        # initialize hidden state to zeros
        h_t = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        for t in range(x.size(1)):  
            h_t = torch.tanh(self.Whx(x[:, t]) + self.Whh(h_t))
        
        out = self.Wph(h_t)

        # y_hat = F.softmax(out, dim=1)
        return out
        
    # add more methods here if needed
