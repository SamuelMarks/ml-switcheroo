import torch


def transpose_matrices(batch):
  """
  Swaps dimensions.
  Semantic pivot: torch.permute -> jax.numpy.transpose
  """
  # Assuming batch of 2D matrices
  return torch.permute(batch, (0, 2, 1))
