"""Test module."""

import ml_switcheroo.frameworks.keras as keras_fw


def test_keras_definitions_hit_true():
  """Test function."""
  # If @property is not evaluating we will override the property on the class just to trigger coverage
  adapter = keras_fw.KerasAdapter()

  # We call the underlying function code manually to cover the branch if it's cached somewhere outside our control
  import ml_switcheroo.frameworks.loader as base

  orig = base.load_definitions
  try:
    base.load_definitions = lambda x: {}
    defs = type(adapter).definitions.fget(adapter)
    assert "ReLU" in defs
  finally:
    base.load_definitions = orig
