"""MNIST CNN model implementation in PyTorch.

This module provides the Net class, which defines a standard 2D Convolutional
Neural Network (CNN) architecture for classifying MNIST digit images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
  """Standard MNIST Convolutional Neural Network model.

  This class implements a simple convolutional neural network optimized
  for the MNIST dataset, consisting of two convolutional layers, max pooling,
  dropout, and two fully connected layers.

  Attributes:
      conv1 (nn.Conv2d): First 2D convolutional layer.
      conv2 (nn.Conv2d): Second 2D convolutional layer.
      dropout1 (nn.Dropout): First dropout layer (probability 0.25).
      dropout2 (nn.Dropout): Second dropout layer (probability 0.50).
      fc1 (nn.Linear): First fully connected layer.
      fc2 (nn.Linear): Second fully connected layer.
  """

  def __init__(self):
    """Initializes the MNIST CNN architecture.

    Configures the layers including two convolutional layers, two dropout layers,
    and two fully connected linear layers.
    """
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)
    self.fc1 = nn.Linear(9216, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward(self, x):
    """Performs a forward pass of the network on the input tensor.

    Args:
        x (torch.Tensor): Input tensor representing a batch of single-channel
            images of shape (N, 1, 28, 28).

    Returns:
        torch.Tensor: Log-softmax probability tensor of shape (N, 10).
    """
    x = self.conv1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    x = self.dropout1(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    output = F.log_softmax(x, dim=1)
    return output
