"""Test suite for the Keras3 module."""

import keras


class FlattenModel:
  """Test double `FlattenModel` implementing `call` for keras3 tests."""

  def __init__(self, start_dim: int = 1) -> None:
    """Initializes the FlattenModel instance.

    Args:
        start_dim: Input `start_dim` parameter value (`int`).
    """
    super().__init__()
    self.flatten = keras.layers.Flatten()

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Execute `flatten` to support call in keras3 tests.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.flatten(x)
