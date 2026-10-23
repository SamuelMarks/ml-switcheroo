"""Test suite for the Tensorflow module."""

import tensorflow as tf


def train_step(
  model: tf.keras.Model,
  optimizer: tf.keras.optimizers.Optimizer,
  x: tf.Tensor,
  y: tf.Tensor,
  loss_fn: tf.Tensor,
) -> tf.Tensor:
  """Trains step.

  Args:
      model: Neural network model instance under test.
      optimizer: The optimizer (tf.keras.optimizers.Optimizer) for train step.
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      loss_fn: The loss fn (tf.Tensor) for train step.

  Returns:
      Resulting `tf.Tensor` from train step.
  """
  with tf.GradientTape() as tape:
    predictions = model(x, training=True)
    loss: tf.Tensor = loss_fn(y, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss
