"""Test suite for the Keras3 module."""

import keras
import tensorflow as tf


def train_step(
  model: keras.Model,
  optimizer: keras.optimizers.Optimizer,
  x: keras.KerasTensor,
  y: keras.KerasTensor,
  loss_fn: keras.KerasTensor,
) -> keras.KerasTensor:
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
    predictions = model(x, training=True)
    loss = loss_fn(y, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss
