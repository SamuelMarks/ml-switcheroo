"""Test suite for the Keras3 module."""

import keras


class Model:
  """Test helper class Model."""

  def __init__(self, in_features: int, out_features: int) -> None:
    """Initializes the Model instance.

    Args:
        in_features: Number of input linear features.
        out_features: Number of output linear features.
    """
    super().__init__()
    self.linear = keras.layers.Dense(out_features, input_dim=in_features)

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    return self.linear(x)
