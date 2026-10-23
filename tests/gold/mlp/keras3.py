"""Test suite for the Keras3 module."""

import keras


class MLP:
  """Test helper class MLP."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
    """Initializes the MLP instance.

    Args:
        in_features: Number of input linear features.
        hidden_features: The hidden features (int) for   init  .
        out_features: Number of output linear features.
    """
    super().__init__()
    self.fc1 = keras.layers.Dense(hidden_features, input_dim=in_features)
    self.fc2 = keras.layers.Dense(out_features)

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    x = self.fc1(x)
    x = keras.activations.relu(x)
    x = self.fc2(x)
    return x
