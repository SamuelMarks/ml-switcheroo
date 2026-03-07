"""Module docstring."""

from praxis import base_layer
import jax.numpy as jnp
import jax


def compute_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # Often defined within task-specific loss functions rather than a standalone op
  from praxis.layers import losses

  # ...
  pass
  # </SWITCHEROO_FAILED_TO_TRANS>
