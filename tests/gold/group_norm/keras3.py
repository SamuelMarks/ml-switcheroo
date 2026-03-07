"""Module docstring."""

import keras


class GNModel(keras.Model):
  """Class docstring."""

  def __init__(self, num_groups: int, num_channels: int):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    self.gn = keras.layers.GroupNormalization(groups=num_groups)
    # </SWITCHEROO_FAILED_TO_TRANS>

  def call(self, x):
    """Function docstring."""
    return self.gn(x)
