"""Test suite for the Paxml module."""

from praxis import base_layer
from praxis.layers import rnn_cell
import jax.numpy as jnp


class LSTMModel(base_layer.BaseLayer):
  """Test suite for the L S T M Model component."""

  input_size: int = 0
  hidden_size: int = 0

  def setup(self):
    """Helper to setup."""
    self.create_child(
      "lstm_cell", rnn_cell.LSTMCellSimple.HParams(num_input_nodes=self.input_size, num_hidden_nodes=self.hidden_size)
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance."""
    pass
