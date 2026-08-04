"""Example module demonstrating array manipulation in JAX.

This module provides utility functions for modifying JAX array layouts,
specifically focusing on dimension permutation operations like transposing
batches of matrices. It serves as a benchmark and example of JAX-to-PyTorch
semantic translation.
"""

import jax.numpy as jnp


def transpose_matrices(batch):
  """Swaps the spatial dimensions of a batch of 2D matrices.

  This function takes a 3D JAX array representing a batch of matrices and
  transposes the spatial height and width axes (axes 1 and 2) while keeping
  the batch dimension (axis 0) intact.

  Semantic pivot:
    jnp.transpose -> torch.permute or torch.transpose.

  Args:
    batch (jax.Array): A 3D JAX array of shape (batch_size, rows, cols)
      representing a batch of 2D matrices.

  Returns:
    jax.Array: A 3D JAX array of shape (batch_size, cols, rows) representing
      the batch of transposed matrices.
  """
  # Assuming batch of 2D matrices
  return jnp.transpose(batch, (0, 2, 1))
