"""Test suite for the Paxml module."""

import typing

import jax.numpy as jnp
from praxis.layers import activations


def gelu_activation(x: jnp.ndarray, approximate: bool = False) -> typing.Any:
  """Helper to gelu activation.

  Args:
      x: Test argument or fixture.
      approximate: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  act = activations.GELU.HParams(approximate=approximate).instantiate()
  return act(x)
