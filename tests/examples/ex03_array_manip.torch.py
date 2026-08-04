"""Example of array and matrix manipulation operations using PyTorch.

This module provides sample functions demonstrating how to permute dimensions and
reshape tensors in PyTorch. It is designed to serve as a reference and testing
example for translating PyTorch tensor operations to JAX equivalent structures.
"""

import torch


def transpose_matrices(batch):
  """Swaps the spatial/inner dimensions of a batch of 2D matrices.

  Semantic pivot: torch.permute -> jax.numpy.transpose.

  Args:
    batch (torch.Tensor): A 3D input tensor representing a batch of 2D matrices,
      typically with shape (batch_size, rows, columns).

  Returns:
    torch.Tensor: A 3D tensor where the last two dimensions are permuted,
      resulting in shape (batch_size, columns, rows).
  """
  # Assuming batch of 2D matrices
  return torch.permute(batch, (0, 2, 1))
