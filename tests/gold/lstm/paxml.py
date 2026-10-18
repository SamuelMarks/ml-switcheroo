"""Test suite for the Paxml module."""

import typing

from praxis.layers import rnn_cell


class LSTMModel:
  """Test helper class LSTMModel."""

  input_size: typing.Any = 0
  hidden_size: typing.Any = 0

  def setup(self: typing.Any) -> None:
    """Helper to setup."""
    self.create_child(
      "lstm_cell", rnn_cell.LSTMCellSimple.HParams(num_input_nodes=self.input_size, num_hidden_nodes=self.hidden_size)
    )

  def __call__(self: typing.Any, x: typing.Any) -> typing.Any:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    pass
