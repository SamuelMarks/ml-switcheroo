"""Example of PyTorch mathematical operations.

This module provides a simple example of calculating the Mean Absolute Error
(MAE) using PyTorch operations. It is used to demonstrate and test the translation
capabilities of mathematical functions from PyTorch to JAX.
"""

import torch


def compute_loss(prediction, target):
  """Calculates Mean Absolute Error.

  Semantic pivot: torch.abs, torch.mean -> jax.numpy.abs, jax.numpy.mean.

  Args:
    prediction (torch.Tensor): The predicted tensor from the model.
    target (torch.Tensor): The ground truth target tensor to compare against.

  Returns:
    torch.Tensor: A scalar tensor representing the Mean Absolute Error loss.
  """
  diff = torch.abs(prediction - target)
  loss = torch.mean(diff)
  return loss
