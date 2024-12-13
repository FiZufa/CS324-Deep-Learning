from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################
class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.Wgx = nn.Linear(input_dim, hidden_dim, bias=True)
        self.Wgh = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.Wix = nn.Linear(input_dim, hidden_dim, bias=True)
        self.Wih = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.Wfx = nn.Linear(input_dim, hidden_dim, bias=True)
        self.Wfh = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.Wox = nn.Linear(input_dim, hidden_dim, bias=True)
        self.Woh = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.Wp = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        batch_size = x.size(0)  
        x = x.view(batch_size, self.seq_length, self.input_dim)  

        h_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        for t in range(self.seq_length):
            x_t = x[:, t, :]
            g_t = torch.tanh(self.Wgx(x_t) + self.Wgh(h_t))
            i_t = torch.sigmoid(self.Wix(x_t) + self.Wih(h_t))
            f_t = torch.sigmoid(self.Wfx(x_t) + self.Wfh(h_t))
            o_t = torch.sigmoid(self.Wox(x_t) + self.Woh(h_t))
            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t

        y = self.Wp(h_t)
        return y
