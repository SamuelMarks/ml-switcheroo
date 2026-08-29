"""Test suite for the Keras3 module."""

import typing

import keras
import tensorflow as tf


def train_step(
  model: keras.Model, optimizer: keras.optimizers.Optimizer, x: typing.Any, y: typing.Any, loss_fn: typing.Any
) -> typing.Any:  # type: ignore
  """Trains step."""
  with tf.GradientTape() as tape:
    predictions = model(x, training=True)
    loss = loss_fn(y, predictions)
  gradients: typing.Any = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # type: ignore
  return loss
