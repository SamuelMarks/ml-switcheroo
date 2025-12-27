"""
Tests for TensorFlow/Keras adapters (Definitions & Wiring).

Verifies:
1. TensorFlow adapter populates Math/Array API mappings.
2. Keras adapter populates Layer/Model mappings.
3. Correct plugin associations (e.g. tf_data_loader).
"""

import pytest
from ml_switcheroo.frameworks.tensorflow import TensorFlowAdapter
from ml_switcheroo.frameworks.keras import KerasAdapter


def test_tf_definitions_content():
  """Verify TF Adapter definitions match expectations."""
  defs = TensorFlowAdapter().definitions

  # 1. Math
  assert defs["Abs"].api == "tf.abs"
  assert defs["Sum"].api == "tf.math.reduce_sum"

  # 2. Extras
  assert defs["DataLoader"].api == "tf.data.Dataset"
  assert defs["DataLoader"].requires_plugin == "tf_data_loader"

  # 3. Permute
  assert defs["permute_dims"].api == "tf.transpose"
  assert defs["permute_dims"].pack_to_tuple == "perm"

  # 4. Neural Layers (TF has no class layers in core adapter, uses Keras or tf.nn functional)
  # Current config maps to None for classes or assumes they aren't mapped
  assert "Linear" not in defs


def test_keras_definitions_content():
  """Verify Keras Adapter definitions."""
  defs = KerasAdapter().definitions

  # 1. Layers
  assert defs["Linear"].api == "keras.layers.Dense"
  assert defs["Linear"].args["out_features"] == "units"

  # 2. Math (Keras Ops)
  assert defs["Abs"].api == "keras.ops.abs"
  assert defs["Add"].api == "keras.ops.add"

  # 3. Model
  assert "Sequential" in defs
  assert defs["Sequential"].requires_plugin == "keras_sequential_pack"


def test_tf_device_syntax():
  """Verify TF device generation logic."""
  adapter = TensorFlowAdapter()
  assert adapter.get_device_syntax("gpu", "0") == "tf.device('GPU:0')"
  assert adapter.get_device_syntax("cpu") == "tf.device('CPU:0')"


def test_keras_discovery_regex():
  """Verify regex patterns for scaffolding."""
  heuristics = KerasAdapter().discovery_heuristics
  assert r"\\.layers\\." in heuristics["neural"]
  assert r"\\.ops\\." in heuristics["array"]
