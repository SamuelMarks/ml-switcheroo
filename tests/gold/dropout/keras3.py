"""Test suite for the Keras3 module."""

import keras


class DropoutModel:
  """Test helper class DropoutModel."""

  def __init__(self, p: float = 0.5) -> None:
    """Initializes the DropoutModel instance.

    Args:
        p: Test argument or fixture.
    """
    super().__init__()
    self.dropout = keras.layers.Dropout(p)

  def call(self, x: keras.KerasTensor, training: bool | None = None) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Test argument or fixture.
        training: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.dropout(x, training=training)
