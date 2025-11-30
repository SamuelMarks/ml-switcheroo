import jax.numpy as jnp


def compute_loss(prediction, target):
  """
  Calculates Mean Absolute Error.
  Semantic pivot: jnp.abs, jnp.mean -> torch.abs, torch.mean
  """
  diff = jnp.abs(prediction - target)
  loss = jnp.mean(diff)
  return loss
