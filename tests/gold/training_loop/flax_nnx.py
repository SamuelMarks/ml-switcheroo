"""Test suite for the Flax Nnx module."""

import typing

import jax.numpy as jnp
from flax import nnx  # type: ignore


def train_step(
  model: nnx.Module, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray, loss_fn: typing.Any
) -> jnp.ndarray:
  """Trains step."""

  def loss_closure(model_ref: nnx.Module) -> jnp.ndarray:
    """Helper to loss closure."""
    predictions = model_ref(x)  # type: ignore
    return typing.cast(jnp.ndarray, loss_fn(predictions, y))

  loss, grads = nnx.value_and_grad(loss_closure)(model)
  optimizer.update(grads)
  return typing.cast(jnp.ndarray, loss)
