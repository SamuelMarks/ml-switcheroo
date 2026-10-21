"""Test suite for the Paxml module."""

from praxis import base_layer
import jax.numpy as jnp

from praxis.layers import rnn_cell


class LSTMModel(base_layer.BaseLayer):
  """Test helper class LSTMModel."""

  input_size = 0
  hidden_size = 0

  def setup(self) -> None:
    """Helper to setup."""
    self.create_child(
      "lstm_cell", rnn_cell.LSTMCellSimple.HParams(num_input_nodes=self.input_size, num_hidden_nodes=self.hidden_size)
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    pass
