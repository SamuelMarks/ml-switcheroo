"""Test suite for the Keras3 module."""

import keras


class TransformerBlock:
  """Test double `TransformerBlock` implementing `call` for keras3 tests."""

  def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1) -> None:
    """Initializes the TransformerBlock instance.

    Args:
        embed_dim: Input `embed_dim` parameter value (`int`).
        num_heads: Input `num_heads` parameter value (`int`).
        ff_dim: Input `ff_dim` parameter value (`int`).
        dropout: Input `dropout` parameter value (`float`).
    """
    super().__init__()

  def call(self, x: keras.KerasTensor, training: bool | None = None) -> keras.KerasTensor:
    """Execute `call` on `TransformerBlock` for keras3 tests.

    Args:
        x: Primary input tensor or array operand.
        training: Input `training` parameter value (`bool | None`).

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
