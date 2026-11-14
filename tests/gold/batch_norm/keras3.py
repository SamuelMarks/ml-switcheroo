"""Test suite for the Keras3 module."""

import keras


class BNModel:
  """Test double `BNModel` implementing `call` for keras3 tests."""

  def __init__(self, num_features: int) -> None:
    """Initializes the BNModel instance.

    Args:
        num_features: Input `num_features` parameter value (`int`).
    """
    super().__init__()
    self.bn = keras.layers.BatchNormalization()

  def call(self, x: keras.KerasTensor, training: bool | None = None) -> keras.KerasTensor:
    """Execute `bn` to support call in keras3 tests.

    Args:
        x: Primary input tensor or array operand.
        training: Input `training` parameter value (`bool | None`).

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.bn(x, training=training)
