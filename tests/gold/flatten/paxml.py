"""Test suite for the Paxml module."""

from praxis import base_layer
import jax.numpy as jnp


class FlattenModel(base_layer.BaseLayer):  # type: ignore[misc]
  """Test helper class FlattenModel."""

  start_dim = 1

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    batch_shape = x.shape[: self.start_dim]
    return x.reshape((*batch_shape, -1))
