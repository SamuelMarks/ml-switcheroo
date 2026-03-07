"""Module docstring."""

import tensorflow as tf


@tf.function
def train_step(
  model: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, x: tf.Tensor, y: tf.Tensor, loss_fn
) -> tf.Tensor:
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  with tf.GradientTape() as tape:
    predictions = model(x, training=True)
    loss = loss_fn(y, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss
  # </SWITCHEROO_FAILED_TO_TRANS>
