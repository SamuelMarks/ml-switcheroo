"""Test suite for the Paxml module."""

import typing
from praxis import base_layer  # type: ignore
from praxis.layers import rnn_cell  # type: ignore


class LSTMModel(base_layer.BaseLayer):  # type: ignore
  """Test suite for the L S T M Model component."""

  input_size: int = 0
  hidden_size: int = 0

  def setup(self) -> None:
    """Helper to setup."""
    self.create_child(
      "lstm_cell", rnn_cell.LSTMCellSimple.HParams(num_input_nodes=self.input_size, num_hidden_nodes=self.hidden_size)
    )

  def __call__(self, x: typing.Any) -> typing.Any:
    """Executes the callable instance."""
    pass
