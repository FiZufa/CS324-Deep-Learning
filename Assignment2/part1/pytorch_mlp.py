from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class MLP(nn.Module):
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__()
        layers = []
        # input layer
        layers.append(nn.Linear(n_inputs, n_hidden[0]))
        layers.append(nn.ReLU())

        # hidden layers
        for i in range(1, len(n_hidden)):
            layers.append(nn.Linear(n_hidden[i-1], n_hidden[i]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(n_hidden[-1], n_classes))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        
        return out
