"""Module docstring."""

import tensorflow as tf
import tensorflow_addons as tfa


class WNModel(tf.keras.Model):
  """Class docstring."""

  def __init__(self, in_features: int, out_features: int):
    """Function docstring."""
    super().__init__()
    # <SWITCHEROO_FAILED_TO_TRANS>
    # TF requires an external Addons wrapper or manual implementation
    self.linear = tfa.layers.WeightNormalization(tf.keras.layers.Dense(out_features))
    # </SWITCHEROO_FAILED_TO_TRANS>

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Function docstring."""
    return self.linear(x)
