"""Test suite for the Keras3 module."""

import keras


class Model:
  """Test helper class Model."""

  def __init__(self, in_features: int, out_features: int) -> None:
    """Initializes the Model instance.

    Args:
        in_features: Test argument or fixture.
        out_features: Test argument or fixture.
    """
    super().__init__()
    self.linear = keras.layers.Dense(out_features, input_dim=in_features)

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.linear(x)
