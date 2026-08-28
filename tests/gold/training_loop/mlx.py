"""Test suite for the Mlx module."""

import typing
import mlx.core as mx  # type: ignore
import mlx.nn as nn  # type: ignore
import mlx.optimizers as optim  # type: ignore


def train_step(model: nn.Module, optimizer: optim.Optimizer, x: mx.array, y: mx.array, loss_fn: typing.Any) -> mx.array:
  """Trains step."""

  def loss_closure(model_ref: typing.Any, x: typing.Any, y: typing.Any) -> typing.Any:
    """Helper to loss closure."""
    return loss_fn(model_ref(x), y)

  loss_and_grad_fn: typing.Any = nn.value_and_grad(model, loss_closure)
  loss, grads = loss_and_grad_fn(model, x, y)
  optimizer.update(model, grads)
  mx.eval(model.parameters(), optimizer.state)
  return typing.cast(mx.array, loss)
