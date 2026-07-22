"""Test suite for the Flax Nnx module."""

from flax import nnx
import jax.numpy as jnp


def train_step(model: nnx.Module, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray, loss_fn) -> jnp.ndarray:
  """Trains step."""

  def loss_closure(model_ref: nnx.Module):
    """Helper to loss closure."""
    predictions = model_ref(x)
    return loss_fn(predictions, y)

  (loss, grads) = nnx.value_and_grad(loss_closure)(model)
  optimizer.update(grads)
  return loss
