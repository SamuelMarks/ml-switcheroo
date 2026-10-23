"""Test suite for the Keras3 module."""

import keras


class GNModel:
  """Test helper class GNModel."""

  def __init__(self, num_groups: int, num_channels: int) -> None:
    """Initializes the GNModel instance.

    Args:
        num_groups: The num groups (int) for   init  .
        num_channels: The num channels (int) for   init  .
    """
    super().__init__()
    self.gn = keras.layers.GroupNormalization(groups=num_groups)

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.gn(x)
