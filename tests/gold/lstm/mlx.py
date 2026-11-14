"""Test suite for the Mlx module."""

import mlx.core as mx
import mlx.nn as nn


class LSTMModel:
  """Sentinel `LSTMModel` fixture class used in MLX tests."""

  def __init__(self, input_size: int, hidden_size: int):
    """Initializes the LSTMModel instance.

    Args:
        input_size: Input `input_size` parameter value (`int`).
        hidden_size: Input `hidden_size` parameter value (`int`).
    """
    super().__init__()
    self.lstm = nn.LSTM(input_size, hidden_size)

  def __call__(self, x: mx.array) -> mx.array:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
    (output, (_hn, _cn)) = self.lstm(x)
    return output
