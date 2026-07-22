"""Test suite for the Paxml module."""

from praxis import optimizers


def setup_adam(model, lr: float = 0.001):
  """Helper to setup adam."""
  return optimizers.Adam.HParams(learning_rate=lr)
