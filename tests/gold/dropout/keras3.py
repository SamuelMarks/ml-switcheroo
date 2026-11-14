"""Test suite for the Keras3 module."""

import keras


class DropoutModel:
  """Test double `DropoutModel` implementing `call` for keras3 tests."""

  def __init__(self, p: float = 0.5) -> None:
    """Initializes the DropoutModel instance.

    Args:
        p: Input `p` parameter value (`float`).
    """
    super().__init__()
    self.dropout = keras.layers.Dropout(p)

  def call(self, x: keras.KerasTensor, training: bool | None = None) -> keras.KerasTensor:
    """Execute `dropout` to support call in keras3 tests.

    Args:
        x: Primary input tensor or array operand.
        training: Input `training` parameter value (`bool | None`).

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.dropout(x, training=training)
