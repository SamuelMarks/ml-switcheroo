"""Module docstring."""

import keras


class FlattenModel(keras.Model):
  """Class docstring."""

  def __init__(self, start_dim: int = 1):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    # Keras Flatten defaults to preserving the first axis (batch)
    # It doesn't natively accept start_dim generically in the constructor
    self.flatten = keras.layers.Flatten()
    # </SWITCHEROO_FAILED_TO_TRANS>

  def call(self, x):
    """Function docstring."""
    return self.flatten(x)
