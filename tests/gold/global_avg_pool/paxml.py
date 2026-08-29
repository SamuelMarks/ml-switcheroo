"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis import base_layer


class GAPModel(base_layer.BaseLayer):
  """Docstring."""

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    return jnp.mean(x, axis=(1, 2))
