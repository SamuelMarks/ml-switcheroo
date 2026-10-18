"""Test suite for the Mlx module."""

import typing

import mlx.nn as _mlx_nn

nn: typing.Any = _mlx_nn


class LSTMModel:
  """Test helper class LSTMModel."""

  def __init__(self, input_size: typing.Any, hidden_size: typing.Any):
    """Initializes the LSTMModel instance.

    Args:
        input_size: Test argument or fixture.
        hidden_size: Test argument or fixture.
    """
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size)

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    (output, (hn, cn)) = self.lstm(x)
    return output
