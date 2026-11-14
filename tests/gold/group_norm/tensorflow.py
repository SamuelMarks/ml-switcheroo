"""TensorFlow golden reference for Group Normalization."""

import tensorflow as tf


class GNModel:
  """TensorFlow ``tf.keras.layers.GroupNormalization`` wrapper."""

  def __init__(self, num_groups: int, num_channels: int) -> None:
    """Initialize TensorFlow GroupNormalization layer.

    Args:
        num_groups: Number of channel groups.
        num_channels: Number of channels ``C``.
    """
    _ = num_channels
    self.gn = tf.keras.layers.GroupNormalization(groups=num_groups)

  def call(self, x: tf.Tensor) -> tf.Tensor:
    """Apply group normalization to ``x``.

    Args:
        x: Input ``tf.Tensor`` operand.

    Returns:
        Normalized ``tf.Tensor``.
    """
    res: tf.Tensor = self.gn(x)
    return res
