"""Module docstring."""

import keras


class BNModel(keras.Model):
  """Class docstring."""

  def __init__(self, num_features: int):
    """Function docstring."""
    super().__init__()
    # Input shape usually inferred or passed implicitly
    self.bn = keras.layers.BatchNormalization()

  def call(self, x, training=None):
    """Function docstring."""
    return self.bn(x, training=training)
