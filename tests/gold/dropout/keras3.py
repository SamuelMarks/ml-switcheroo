"""Module docstring."""

import keras


class DropoutModel(keras.Model):
  """Class docstring."""

  def __init__(self, p: float = 0.5):
    """Function docstring."""
    super().__init__()
    self.dropout = keras.layers.Dropout(p)

  def call(self, x, training=None):
    """Function docstring."""
    # <SWITCHEROO_FAILED_TO_TRANS>
    # Keras expects an explicit training flag in the call signature
    return self.dropout(x, training=training)
    # </SWITCHEROO_FAILED_TO_TRANS>
