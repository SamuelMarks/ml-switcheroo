"""Module docstring."""

import keras


class Model(keras.Model):
  """Class docstring."""

  def __init__(self, in_features: int, out_features: int):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    # Keras 3 often infers input shape by default, but dense maps to Dense
    self.linear = keras.layers.Dense(out_features, input_dim=in_features)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def call(self, x):
    """Function docstring."""
    return self.linear(x)
