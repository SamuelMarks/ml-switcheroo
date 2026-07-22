"""Test suite for the Keras3 module."""

import keras


class MLP(keras.Model):
  """Test suite for the M L P component."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int):
    """Initializes the MLP instance."""
    super().__init__()
    self.fc1 = keras.layers.Dense(hidden_features, input_dim=in_features)
    self.fc2 = keras.layers.Dense(out_features)

  def call(self, x):
    """Helper to call."""
    x = self.fc1(x)
    x = keras.activations.relu(x)
    x = self.fc2(x)
    return x
