"""Test suite for the Keras3 module."""

import keras


class DropoutModel:
  """Test helper class DropoutModel."""

  def __init__(self, p: float = 0.5) -> None:
    """Initializes the DropoutModel instance.

    Args:
        p: The p (float) for   init  .
    """
    super().__init__()
    self.dropout = keras.layers.Dropout(p)

  def call(self, x: keras.KerasTensor, training: bool | None = None) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.
        training: The training (bool | None) for call.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.dropout(x, training=training)
