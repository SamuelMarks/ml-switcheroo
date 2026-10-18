"""Test suite for the Paxml module."""

import typing

from praxis import optimizers


def setup_adam(model: typing.Any, lr: float = 0.001) -> typing.Any:
  """Helper to setup adam.

  Args:
      model: Test argument or fixture.
      lr: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return optimizers.Adam.HParams(learning_rate=lr)
