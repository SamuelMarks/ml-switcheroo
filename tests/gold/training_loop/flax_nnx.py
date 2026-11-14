"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


def train_step(
  model: nnx.Module, optimizer: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, loss_fn: jnp.ndarray
) -> jnp.ndarray:
  """Trains step.

  Args:
      model: Neural network model instance under test.
      optimizer: Input `optimizer` parameter value (`jnp.ndarray`).
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      loss_fn: Input `loss_fn` parameter value (`jnp.ndarray`).

  Returns:
      Resulting `jnp.ndarray` from train step.
  """

  def loss_closure(model_ref: jnp.ndarray) -> jnp.ndarray:
    """Execute `model_ref` to support loss closure in flax NNX tests.

    Args:
        model_ref: Input `model_ref` parameter value (`jnp.ndarray`).

    Returns:
        Resulting `jnp.ndarray` from loss closure.
    """
    predictions = model_ref(x)
    return loss_fn(predictions, y)

  loss, grads = nnx.value_and_grad(loss_closure)(model)
  optimizer.update(grads)
  return loss
