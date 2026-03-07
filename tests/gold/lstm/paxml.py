"""Module docstring."""

from praxis import base_layer
from praxis.layers import rnn_cell
import jax.numpy as jnp


class LSTMModel(base_layer.BaseLayer):
  """Class docstring."""

  input_size: int = 0
  hidden_size: int = 0

  def setup(self):
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # PaxML typically uses stacked RNN cells and a separate loop mechanism
    self.create_child(
      "lstm_cell", rnn_cell.LSTMCellSimple.HParams(num_input_nodes=self.input_size, num_hidden_nodes=self.hidden_size)
    )
    # </SWITCHEROO_FAILED_TO_TRANS>

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Function docstring."""
    pass
