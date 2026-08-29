"""Test suite for the Tensorflow module."""

import typing

import tensorflow as tf


class DropoutModel(tf.keras.Model):  # type: ignore
  """Docstring."""

  def __init__(self, p: float = 0.5) -> None:
    """Initializes the DropoutModel instance."""
    super().__init__()
    self.dropout: typing.Any = tf.keras.layers.Dropout(p)

  def call(self, x: typing.Any, training: typing.Optional[bool] = None) -> typing.Any:
    """Helper to call."""
    return self.dropout(x, training=training)
