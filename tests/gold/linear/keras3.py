"""Test suite for the Keras3 module."""

import keras


class Model(keras.Model):
  """Test suite for the Model component."""

  def __init__(self, in_features: int, out_features: int):
    """Initializes the Model instance."""
    super().__init__()
    self.linear = keras.layers.Dense(out_features, input_dim=in_features)

  def call(self, x):
    """Helper to call."""
    return self.linear(x)
