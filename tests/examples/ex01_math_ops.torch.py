import torch


def compute_loss(prediction, target):
  """
  Calculates Mean Absolute Error.
  Semantic pivot: torch.abs, torch.mean -> jax.numpy.abs, jax.numpy.mean
  """
  diff = torch.abs(prediction - target)
  loss = torch.mean(diff)
  return loss
