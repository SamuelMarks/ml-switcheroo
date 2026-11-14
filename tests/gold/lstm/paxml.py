"""Test suite for the Paxml module."""

import jax.numpy as jnp
from praxis import base_layer
from praxis.layers import rnn_cell


class LSTMModel(base_layer.BaseLayer):
  """Stub `base_layer.BaseLayer` subclass implementing `setup` for PaxML tests."""

  input_size = 0
  hidden_size = 0

  def setup(self) -> None:
    """Execute `create_child` to support setup in PaxML tests."""
    self.create_child(
      "lstm_cell", rnn_cell.LSTMCellSimple.HParams(num_input_nodes=self.input_size, num_hidden_nodes=self.hidden_size)
    )

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Executes the callable instance.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Output tensor or value computed by the forward pass.
    """
