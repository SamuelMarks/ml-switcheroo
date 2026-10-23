"""Test suite for the Keras3 module."""

import keras


class BNModel:
  """Test helper class BNModel."""

  def __init__(self, num_features: int) -> None:
    """Initializes the BNModel instance.

    Args:
        num_features: The num features (int) for   init  .
    """
    super().__init__()
    self.bn = keras.layers.BatchNormalization()

  def call(self, x: keras.KerasTensor, training: bool | None = None) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.
        training: The training (bool | None) for call.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.bn(x, training=training)
