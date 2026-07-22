"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


def train_step(model: nn.Module, optimizer: optim.Optimizer, x: mx.array, y: mx.array, loss_fn) -> mx.array:
  """Trains step."""

  def loss_closure(model_ref, x, y):
    """Helper to loss closure."""
    return loss_fn(model_ref(x), y)

  loss_and_grad_fn = nn.value_and_grad(model, loss_closure)
  (loss, grads) = loss_and_grad_fn(model, x, y)
  optimizer.update(model, grads)
  mx.eval(model.parameters(), optimizer.state)
  return loss
