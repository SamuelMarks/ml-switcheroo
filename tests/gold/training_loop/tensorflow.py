"""Test suite for the Tensorflow module."""

import typing

import tensorflow as tf


def train_step(
  model: tf.keras.Model,
  optimizer: tf.keras.optimizers.Optimizer,
  x: tf.Tensor,
  y: tf.Tensor,
  loss_fn: typing.Any,
) -> tf.Tensor:
  """Trains step.

  Args:
      model: Test argument or fixture.
      optimizer: Test argument or fixture.
      x: Test argument or fixture.
      y: Test argument or fixture.
      loss_fn: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  with tf.GradientTape() as tape:
    predictions: typing.Any = model(x, training=True)
    loss: tf.Tensor = loss_fn(y, predictions)
  gradients: typing.Any = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss
