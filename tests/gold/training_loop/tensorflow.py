"""Test suite for the Tensorflow module."""

import typing
import tensorflow as tf


@tf.function  # type: ignore
def train_step(
  model: tf.keras.Model,
  optimizer: tf.keras.optimizers.Optimizer,
  x: tf.Tensor,
  y: tf.Tensor,
  loss_fn: typing.Any,  # type: ignore
) -> tf.Tensor:
  """Trains step."""
  with tf.GradientTape() as tape:
    predictions = model(x, training=True)
    loss: tf.Tensor = loss_fn(y, predictions)
  gradients: typing.Any = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # type: ignore
  return loss
