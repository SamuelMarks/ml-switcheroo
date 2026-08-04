"""Example of a simple convolutional neural network built using Keras.

This module defines a basic CNN model via the Keras Functional API along with
a custom mean squared error loss function using backend-agnostic Keras ops.
It is designed to demonstrate and test Keras transpilation capabilities.
"""

import keras
from keras import layers, ops


def build_model(input_shape, num_classes):
  """Builds a Keras Functional API CNN model for classification.

  Source: Keras Examples.

  Args:
    input_shape (tuple of int): A tuple of integers representing the input shape,
      excluding the batch size dimension.
    num_classes (int): The number of target output classes for classification.

  Returns:
    keras.Model: A Keras Functional Model representing the CNN classifier.
  """
  inputs = keras.Input(shape=input_shape)
  x = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
  x = layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = layers.Flatten()(x)
  x = layers.Dropout(0.5)(x)
  outputs = layers.Dense(num_classes, activation="softmax")(x)
  return keras.Model(inputs, outputs)


def custom_loss(y_true, y_pred):
  """Computes the mean squared error loss using backend-agnostic Keras ops.

  Args:
    y_true (Any): The ground-truth target values/labels.
    y_pred (Any): The predicted outputs from the model.

  Returns:
    Any: A scalar tensor representing the mean squared error loss.
  """
  # Using backend-agnostic ops
  return ops.mean(ops.square(y_true - y_pred))
