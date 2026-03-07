"""Module docstring."""

import keras


class LayerNormModel(keras.Model):
  """Class docstring."""

  def __init__(self, normalized_shape: int):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    # Keras LayerNormalization doesn't explicitly take normalized_shape in the constructor typically
    self.ln = keras.layers.LayerNormalization(axis=-1)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def call(self, x):
    """Function docstring."""
    return self.ln(x)
