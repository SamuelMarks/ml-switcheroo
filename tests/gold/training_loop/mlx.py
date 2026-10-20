"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


def train_step(model: mx.array, optimizer: mx.array, x: mx.array, y: mx.array, loss_fn: mx.array) -> mx.array:
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

  def loss_closure(model_ref: mx.array, x: mx.array, y: mx.array) -> mx.array:
    """Helper to loss closure.

    Args:
        model_ref: Test argument or fixture.
        x: Test argument or fixture.
        y: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return loss_fn(model_ref(x), y)

  loss_and_grad_fn = nn.value_and_grad(model, loss_closure)
  loss, grads = loss_and_grad_fn(model, x, y)
  optimizer.update(model, grads)
  mx.eval(model.parameters(), optimizer.state)
  return loss
