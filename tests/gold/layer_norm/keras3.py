"""Test suite for the Keras3 module."""

import keras


class LayerNormModel:
  """Test helper class LayerNormModel."""

  def __init__(self, normalized_shape: int) -> None:
    """Initializes the LayerNormModel instance.

    Args:
        normalized_shape: The normalized shape (int) for   init  .
    """
    super().__init__()
    self.ln = keras.layers.LayerNormalization(axis=-1)

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.ln(x)
