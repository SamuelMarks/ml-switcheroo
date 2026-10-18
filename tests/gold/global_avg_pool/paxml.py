"""Test suite for the Paxml module."""

import typing

import jax.numpy as jnp


class GAPModel:
  """Test helper class GAPModel."""

  def __call__(self: typing.Any, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return jnp.mean(x, axis=(1, 2))
