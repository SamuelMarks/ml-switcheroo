"""Test module."""

import sys
import importlib
from unittest.mock import patch


def test_keras_import_error():
  """Test function."""
  with patch.dict(sys.modules, {"keras": None, "keras.activations": None}):
    import ml_switcheroo.frameworks.keras as keras_fw

    importlib.reload(keras_fw)
    assert keras_fw.keras is None
  importlib.reload(keras_fw)
