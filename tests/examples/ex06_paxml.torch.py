"""Example of a simple Multi-Layer Perceptron built using PyTorch for PaxML.

This module defines a basic MLP structure designed to demonstrate and verify the
capabilities of translating PyTorch structures into PaxML (Praxis-based) layers.
"""

import torch.nn as nn


class SimpleMLP(nn.Module):
  """A simple Multi-Layer Perceptron for PaxML conversion testing.

  Semantic pivot:
    - nn.Module -> praxis.base_layer.BaseLayer
    - __init__ -> setup
    - forward -> __call__
    - nn.Linear -> praxis.layers.Linear.
  """

  def __init__(self, input_size, hidden_size, num_classes):
    """Initializes the SimpleMLP model with specified layer sizes.

    Args:
      input_size (int): The dimensionality of the input features.
      hidden_size (int): The number of hidden units in the intermediate layer.
      num_classes (int): The number of output classes/features.
    """
    super().__init__()
    # Standard Linear layer
    self.fc1 = nn.Linear(input_size, hidden_size)
    # Activation
    self.relu = nn.ReLU()
    # Output layer
    self.fc2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    """Computes the forward pass of the MLP on the input tensor.

    Args:
      x (torch.Tensor): The input feature tensor of shape (batch_size, input_size).

    Returns:
      torch.Tensor: The output activation/logit tensor of shape (batch_size, num_classes).
    """
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out
