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
      model: Neural network model instance under test.
      optimizer: Input `optimizer` parameter value (`keras.optimizers.Optimizer`).
      x: Primary input tensor or array operand.
      y: Secondary input tensor or target operand.
      loss_fn: Input `loss_fn` parameter value (`keras.KerasTensor`).

  Returns:
      Resulting `keras.KerasTensor` from train step.
  """
  with tf.GradientTape() as tape:
    predictions = model(x, training=True)
    loss = loss_fn(y, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables, strict=False))
  return loss
