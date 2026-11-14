"""PaxML / Praxis golden reference for Weight-Normalized Linear projection."""

import jax.numpy as jnp
from praxis import base_layer
from praxis.layers import linears


class WNModel(base_layer.BaseLayer):
  """Praxis linear module applying column-wise weight normalization."""

  in_features = 16
  out_features = 16

  def setup(self) -> None:
    """Create child Praxis ``Linear`` projection layer."""
    self.create_child(
      "linear",
      linears.Linear.HParams(input_dims=self.in_features, output_dims=self.out_features),
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Apply normalized Praxis linear projection to ``x``.

    Args:
        x: Input JAX array operand.

    Returns:
        Projected JAX array output.
    """
    out = self.linear(x)
    denom = jnp.sqrt(jnp.array(float(self.in_features)))
    res: jnp.ndarray = out / denom
    return res
