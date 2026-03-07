"""Module docstring."""

import keras


def gelu_activation(x, approximate: bool = False):
  """Function docstring."""
  # <SWITCHEROO_FAILED_TO_TRANS>
  # Keras 3 uses boolean approximate
  return keras.activations.gelu(x, approximate=approximate)
  # </SWITCHEROO_FAILED_TO_TRANS>
