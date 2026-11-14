"""Test suite for the Keras3 module."""

import keras


class MLP:
  """Test double `MLP` implementing `call` for keras3 tests."""

  def __init__(self, in_features: int, hidden_features: int, out_features: int) -> None:
    """Initializes the MLP instance.

    Args:
        in_features: Number of input linear features.
        hidden_features: Input `hidden_features` parameter value (`int`).
        out_features: Number of output linear features.
    """
    super().__init__()
    self.fc1 = keras.layers.Dense(hidden_features, input_dim=in_features)
    self.fc2 = keras.layers.Dense(out_features)

  def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
    """Execute `fc1` to support call in keras3 tests.

    Args:
        x: Primary input tensor or array operand.

    Returns:
        Resulting `keras.KerasTensor` from call.
    """
    x = self.fc1(x)
    x = keras.activations.relu(x)
    return self.fc2(x)
