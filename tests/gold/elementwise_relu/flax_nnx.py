"""Test suite for the Flax Nnx module."""

import typing

import jax
import jax.numpy as jnp


def relu_activation(x: jnp.ndarray) -> typing.Any:
  """Helper to relu activation.

  Args:
      x: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return jax.nn.relu(x)
