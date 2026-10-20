"""Test suite for the Flax Nnx module."""

import jax.numpy as jnp
from flax import nnx


def train_step(
  model: nnx.Module, optimizer: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, loss_fn: jnp.ndarray
) -> jnp.ndarray:
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

  def loss_closure(model_ref: jnp.ndarray) -> jnp.ndarray:
    """Helper to loss closure.

    Args:
        model_ref: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    predictions = model_ref(x)
    return loss_fn(predictions, y)

  loss, grads = nnx.value_and_grad(loss_closure)(model)
  optimizer.update(grads)
  return loss
