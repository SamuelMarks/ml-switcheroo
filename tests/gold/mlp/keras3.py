"""Test suite for the Keras3 module."""

import typing

import keras


class MLP(keras.Model):  # type: ignore
  """Docstring."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
    """Initializes the MLP instance."""
    super().__init__()
    self.fc1: typing.Any = keras.layers.Dense(hidden_features, input_dim=in_features)  # type: ignore
    self.fc2: typing.Any = keras.layers.Dense(out_features)  # type: ignore

  def call(self, x: typing.Any) -> typing.Any:
    """Helper to call."""
    x = self.fc1(x)
    x = keras.activations.relu(x)  # type: ignore
    x = self.fc2(x)
    return x
