"""Test suite for the Keras3 module."""

import keras


class GNModel:
  """Test double `GNModel` implementing `call` for keras3 tests."""

  def __init__(self, num_groups: int, num_channels: int) -> None:
    """Initializes the GNModel instance.

    Args:
        num_groups: Input `num_groups` parameter value (`int`).
        num_channels: Input `num_channels` parameter value (`int`).
    """
    super().__init__()
    self.gn = keras.layers.GroupNormalization(groups=num_groups)

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Execute `gn` to support call in keras3 tests.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.gn(x)
