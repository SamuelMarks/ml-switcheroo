"""Test suite for the Keras3 module."""

import keras


class DropoutModel(keras.Model):
  """Test suite for the Dropout Model component."""

  def __init__(self, p: float = 0.5):
    """Initializes the DropoutModel instance."""
    super().__init__()
    self.dropout = keras.layers.Dropout(p)

  def call(self, x, training=None):
    """Helper to call."""
    return self.dropout(x, training=training)
