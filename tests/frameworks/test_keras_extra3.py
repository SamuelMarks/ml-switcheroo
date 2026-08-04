"""Test module."""

from unittest.mock import patch
import ml_switcheroo.frameworks.keras as keras_fw


def test_keras_definitions_no_mock_hit_208():
  """Test function."""
  import importlib

  from ml_switcheroo.frameworks.loader import load_definitions

  if hasattr(load_definitions, "cache_clear"):
    load_definitions.cache_clear()

  with patch("ml_switcheroo.frameworks.loader.json.load") as mock_json:
    mock_json.return_value = {}
    importlib.reload(keras_fw)
    adapter = keras_fw.KerasAdapter()
    defs = adapter.definitions
    assert "ReLU" in defs

  if hasattr(load_definitions, "cache_clear"):
    load_definitions.cache_clear()

  with patch("ml_switcheroo.frameworks.loader.json.load") as mock_json:
    mock_json.return_value = {"ReLU": {"api": "keras.layers.ReLU"}}
    importlib.reload(keras_fw)
    adapter2 = keras_fw.KerasAdapter()
    defs2 = adapter2.definitions
    assert defs2["ReLU"].api == "keras.layers.ReLU"
