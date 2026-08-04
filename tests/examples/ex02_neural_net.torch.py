"""Example of a simple neural network built using PyTorch.

This module defines a basic single-layer perceptron model which is utilized for
demonstrating and testing translation capabilities from PyTorch to JAX/Flax structures.
"""

import torch.nn as nn


class SimplePerceptron(nn.Module):
  """Basic Single-Layer Perceptron.

  This class implements a simple single-layer perceptron model in PyTorch, consisting
  of a single linear transformation layer. It is used as a basic model for translation.

  Semantic pivot:
    - nn.Module -> flax.nnx.Module
    - nn.Linear -> flax.nnx.Linear
    - forward -> __call__.

  Attributes:
    layer (nn.Linear): The linear transformation layer.
  """

  def __init__(self, in_features, out_features):
    """Initializes the SimplePerceptron layer.

    Args:
      in_features (int): The number of input features/dimensions.
      out_features (int): The number of output features/dimensions.
    """
    super().__init__()
    # Standard Linear layer
    self.layer = nn.Linear(in_features, out_features)

  def forward(self, x):
    """Performs a forward pass of the neural network layer.

    Args:
      x (torch.Tensor): The input tensor to transform.

    Returns:
      torch.Tensor: The output tensor resulting from the linear transformation.
    """
    return self.layer(x)
