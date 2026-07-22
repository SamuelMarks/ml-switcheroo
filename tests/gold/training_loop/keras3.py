"""Test suite for the Keras3 module."""

import keras
import tensorflow as tf


def train_step(model: keras.Model, optimizer: keras.optimizers.Optimizer, x, y, loss_fn):
  """Trains step."""
  with tf.GradientTape() as tape:
    predictions = model(x, training=True)
    loss = loss_fn(y, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return loss
