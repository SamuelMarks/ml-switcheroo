"""Test suite for the Keras Init module."""

import ml_switcheroo.frameworks.keras as keras_fw


def test_keras_init_no_keras():
  """Verifies the behavior of Keras initialization no Keras."""
  original_keras = keras_fw.keras
  keras_fw.keras = None
  try:
    adapter = keras_fw.KerasAdapter()
    assert adapter._mode == keras_fw.InitMode.GHOST
  finally:
    keras_fw.keras = original_keras
