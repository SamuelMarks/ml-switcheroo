"""Test suite for the Keras3 module."""

import keras


class MLP:
  """Test helper class MLP."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
    """Initializes the MLP instance.

    Args:
        in_features: Test argument or fixture.
        hidden_features: Test argument or fixture.
        out_features: Test argument or fixture.
    """
    super().__init__()
    self.fc1 = keras.layers.Dense(hidden_features, input_dim=in_features)
    self.fc2 = keras.layers.Dense(out_features)

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Helper to call.

    Args:
        x: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    x = self.fc1(x)
    x = keras.activations.relu(x)
    x = self.fc2(x)
    return x
