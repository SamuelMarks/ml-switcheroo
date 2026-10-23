"""Test suite for the Keras3 module."""

import keras


class MaxPoolModel:
  """Test helper class MaxPoolModel."""

  def __init__(self, pool_size: int = 2, strides: int = 2) -> None:
    """Initializes the MaxPoolModel instance.

    Args:
        pool_size: The pool size (int) for   init  .
        strides: The strides (int) for   init  .
    """
    super().__init__()
    self.pool = keras.layers.MaxPooling2D(pool_size=pool_size, strides=strides)

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.pool(x)
