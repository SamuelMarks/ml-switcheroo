"""Test suite for the Flax Nnx module."""

import jax
import jax.numpy as jnp


def gelu_activation(x: jnp.ndarray, approximate: bool = False) -> jnp.ndarray:
  """Execute `gelu` to support gelu activation in flax NNX tests.

  Args:
      x: Primary input tensor or array operand.
      approximate: Input `approximate` parameter value (`bool`).

  Returns:
      Resulting `jnp.ndarray` from gelu activation.
  """
  return jax.nn.gelu(x, approximate=approximate)
