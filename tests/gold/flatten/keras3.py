"""Test suite for the Keras3 module."""

import keras


class FlattenModel:
  """Test helper class FlattenModel."""

  def __init__(self, start_dim: int = 1) -> None:
    """Initializes the FlattenModel instance.

    Args:
        start_dim: The start dim (int) for   init  .
    """
    super().__init__()
    self.flatten = keras.layers.Flatten()

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.flatten(x)
