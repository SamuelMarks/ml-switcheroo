"""Test suite for the Keras3 module."""

import keras


class LayerNormModel(keras.Model):
  """Test suite for the Layer Norm Model component."""

  def __init__(self, normalized_shape: int):
    """Initializes the LayerNormModel instance."""
    super().__init__()
    self.ln = keras.layers.LayerNormalization(axis=-1)

  def call(self, x):
    """Helper to call."""
    return self.ln(x)
