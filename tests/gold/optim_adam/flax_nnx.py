"""Module docstring."""

from flax import nnx
import optax


def setup_adam(model: nnx.Module, lr: float = 0.001):
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # NNX uses optax
  optimizer = nnx.Optimizer(model, optax.adam(lr))
  return optimizer
  # </SWITCHEROO_FAILED_TO_TRANS>
