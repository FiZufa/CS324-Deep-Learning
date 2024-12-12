from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

  def __init__(self, n_channels, n_classes):
    """
    Initializes CNN object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    """

    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=1)
    self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2), padding=1)
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=1)
    self.pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
    self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)
    self.pool3 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
    self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=1)
    self.fc1 = nn.Linear(4*4*512, 256)
    self.fc2 = nn.Linear(256, 64)
    self.fc3 = nn.Linear(64,10)


  def forward(self, x):
    """
    Performs forward pass of the input.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """

    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))
    x = F.relu(self.conv3(x))
    x = self.pool3(F.relu(self.conv4(x)))

    x = x.view(-1, 4*4 * 512)

    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.softmax(self.fc3(x), dim=0)

    return x
