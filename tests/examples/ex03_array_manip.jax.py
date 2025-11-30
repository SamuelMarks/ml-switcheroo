import jax.numpy as jnp


def transpose_matrices(batch):
  """
  Swaps dimensions.
  Semantic pivot: jnp.transpose -> torch.permute or torch.transpose
  """
  # Assuming batch of 2D matrices
  return jnp.transpose(batch, (0, 2, 1))
