"""Test suite for the Keras3 module."""

import keras


class LayerNormModel:
  """Test double `LayerNormModel` implementing `call` for keras3 tests."""

  def __init__(self, normalized_shape: int) -> None:
    """Initializes the LayerNormModel instance.

    Args:
        normalized_shape: Input `normalized_shape` parameter value (`int`).
    """
    super().__init__()
    self.ln = keras.layers.LayerNormalization(axis=-1)

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Execute `ln` to support call in keras3 tests.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.ln(x)
