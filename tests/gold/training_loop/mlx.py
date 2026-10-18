"""Test suite for the Mlx module."""

import typing

import mlx.core as mx
import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


def train_step(model: typing.Any, optimizer: typing.Any, x: typing.Any, y: typing.Any, loss_fn: typing.Any) -> typing.Any:
  """Trains step.

  Args:
      model: Test argument or fixture.
      optimizer: Test argument or fixture.
      x: Test argument or fixture.
      y: Test argument or fixture.
      loss_fn: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """

  def loss_closure(model_ref: typing.Any, x: typing.Any, y: typing.Any) -> typing.Any:
    """Helper to loss closure.

    Args:
        model_ref: Test argument or fixture.
        x: Test argument or fixture.
        y: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return loss_fn(model_ref(x), y)

  loss_and_grad_fn: typing.Any = nn.value_and_grad(model, loss_closure)
  loss, grads = loss_and_grad_fn(model, x, y)
  optimizer.update(model, grads)
  mx.eval(model.parameters(), optimizer.state)
  return typing.cast(mx.array, loss)
