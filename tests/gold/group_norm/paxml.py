"""PaxML / Praxis golden reference for Group Normalization."""

import jax.numpy as jnp
from praxis import base_layer
from praxis.layers import normalizations


class GNModel(base_layer.BaseLayer):
  """Praxis ``GroupNorm`` wrapper module."""

  num_groups = 1
  num_channels = 1

  def setup(self) -> None:
    """Instantiate Praxis ``GroupNorm`` child layer."""
    self.create_child(
      "gn",
      normalizations.GroupNorm.HParams(
        num_groups=self.num_groups,
        dim=self.num_channels,
      ),
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Apply Praxis group normalization to ``x``.

    Args:
        x: Input JAX array operand.

    Returns:
        Group-normalized JAX array.
    """
    res: jnp.ndarray = self.gn(x)
    return res
