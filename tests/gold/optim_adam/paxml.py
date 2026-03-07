"""Module docstring."""

from praxis import optimizers


def setup_adam(model, lr: float = 0.001):
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  return optimizers.Adam.HParams(learning_rate=lr)
  # </SWITCHEROO_FAILED_TO_TRANS>
