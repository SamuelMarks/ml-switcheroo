"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


def train_step(model: mx.array, optimizer: mx.array, x: mx.array, y: mx.array, loss_fn: mx.array) -> mx.array:
  """Trains step.

  Args:
      model: Neural network model instance under test.
      optimizer: The optimizer (mx.array) for train step.
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      loss_fn: The loss fn (mx.array) for train step.

  Returns:
      Resulting `mx.array` from train step.
  """

  def loss_closure(model_ref: mx.array, x: mx.array, y: mx.array) -> mx.array:
    """Helper to loss closure.

    Args:
        model_ref: The model ref (mx.array) for loss closure.
        x: Primary input tensor or array operand.
        y: Secondary input tensor or target operand.

    Returns:
        Resulting `mx.array` from loss closure.
    """
    return loss_fn(model_ref(x), y)

  loss_and_grad_fn = nn.value_and_grad(model, loss_closure)
  loss, grads = loss_and_grad_fn(model, x, y)
  optimizer.update(model, grads)
  mx.eval(model.parameters(), optimizer.state)
  return loss
