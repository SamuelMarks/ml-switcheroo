"""Module docstring."""

import keras


class MLP(keras.Model):
  """Class docstring."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int):
    """Function docstring."""
    super().__init__()
    self.fc1 = keras.layers.Dense(hidden_features, input_dim=in_features)
    self.fc2 = keras.layers.Dense(out_features)

  def call(self, x):
    """Function docstring."""
    x = self.fc1(x)
    x = keras.activations.relu(x)
    x = self.fc2(x)
    return x
