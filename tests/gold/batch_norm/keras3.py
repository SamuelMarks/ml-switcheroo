"""Test suite for the Keras3 module."""

import keras


class BNModel(keras.Model):
  """Docstring."""

  def __init__(self, num_features: int):
    """Initializes the BNModel instance."""
    super().__init__()
    self.bn = keras.layers.BatchNormalization()

  def call(self, x, training=None):
    """Helper to call."""
    return self.bn(x, training=training)
