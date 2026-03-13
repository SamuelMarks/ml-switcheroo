import sys
import pytest
from ml_switcheroo.frameworks.keras import KerasAdapter
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.base import StandardCategory
from ml_switcheroo.core.ghost import GhostRef


def test_keras_adapter():
  adapter = KerasAdapter()
  assert adapter.display_name == "Keras"
  assert adapter.ui_priority == 25
  assert adapter.inherits_from is None

  assert (
    adapter.search_modules == ["keras.ops", "keras.layers", "keras.activations", "keras.random"]
    or adapter.search_modules == []
  )
  assert adapter.unsafe_submodules == set()
  assert adapter.import_alias == ("keras", "keras")

  ns = adapter.import_namespaces
  assert "keras" in ns
  assert ns["keras"].tier == SemanticTier.NEURAL
  assert ns["keras.ops"].tier == SemanticTier.ARRAY_API

  heuristics = adapter.discovery_heuristics
  assert "neural" in heuristics
  assert "array" in heuristics

  tc = adapter.test_config
  assert "import" in tc

  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert adapter.get_to_numpy_code() != ""
  assert SemanticTier.ARRAY_API in adapter.supported_tiers
  assert adapter.declared_magic_args == []

  traits = adapter.structural_traits
  assert traits.module_base == "keras.Layer"

  pt = adapter.plugin_traits
  assert pt.has_numpy_compatible_arrays

  defs = adapter.definitions
  assert "ReLU" in defs

  assert "utils.set_random_seed" in adapter.rng_seed_methods

  from ml_switcheroo.frameworks.base import InitMode

  # Discovery
  adapter._mode = InitMode.LIVE
  from unittest.mock import patch, MagicMock
  import types

  mock_keras = types.ModuleType("keras")
  mock_keras.ops = types.ModuleType("keras.ops")
  mock_keras.layers = types.ModuleType("keras.layers")
  mock_keras.losses = types.ModuleType("keras.losses")
  mock_keras.optimizers = types.ModuleType("keras.optimizers")
  mock_keras.activations = types.ModuleType("keras.activations")

  with patch("ml_switcheroo.frameworks.keras.keras", mock_keras):
    # To make it scan something
    with patch("ml_switcheroo.frameworks.keras.inspect.getmembers", return_value=[("relu", lambda x: x)]):
      refs = adapter.collect_api(StandardCategory.ACTIVATION)
      assert isinstance(refs, list)

      refs = adapter.collect_api(StandardCategory.LAYER)
      assert isinstance(refs, list)

      refs = adapter.collect_api(StandardCategory.LOSS)
      assert isinstance(refs, list)

      refs = adapter.collect_api(StandardCategory.OPTIMIZER)
      assert isinstance(refs, list)

  # Test convert
  mock_keras.ops.convert_to_tensor = MagicMock(return_value="tensor")
  with patch.dict("sys.modules", {"keras": mock_keras}):
    res = adapter.convert([1, 2, 3])
    assert res == "tensor"

  res = adapter.convert("test")
  assert res == "test"

  # Serialization
  assert adapter.get_serialization_imports() == ["import keras"]
  assert adapter.get_serialization_syntax("save", "'file.h5'", "model") == "model.save('file.h5')"
  assert adapter.get_serialization_syntax("load", "'file.h5'") == "keras.saving.load_model('file.h5')"
  assert adapter.get_serialization_syntax("unknown", "'file.h5'") == ""

  assert adapter.get_weight_conversion_imports() == ["import keras", "import numpy as np", "import h5py"]
  assert "keras.models.load_model(path" in adapter.get_weight_load_code("path")
  assert "tensor.numpy()" in adapter.get_tensor_to_numpy_expr("tensor")
  assert "h5py.File(path" in adapter.get_weight_save_code("state", "path")

  assert "keras.name_scope('gpu')" in adapter.get_device_syntax("cuda")
  assert "keras.name_scope('cpu')" in adapter.get_device_syntax("cpu")

  assert adapter.get_device_check_syntax() != ""
  assert adapter.get_rng_split_syntax("r", "k") == "pass"

  adapter.apply_wiring({})
  assert adapter.get_doc_url("keras.layers.Dense") == "https://keras.io/search.html?q=keras.layers.Dense"

  examples = adapter.get_tiered_examples()
  assert "tier1_math" in examples
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples

  # Scan module with empty list
  refs = adapter._scan_module(None, "prefix", "class")
  assert refs == []

  # Test convert exception handling if import fails
  import sys

  old_keras = sys.modules.get("keras")
  sys.modules["keras"] = None
  try:
    assert adapter.convert([1]) == [1]
  finally:
    sys.modules["keras"] = old_keras

  adapter._mode = InitMode.GHOST
  refs = adapter.collect_api(StandardCategory.ACTIVATION)
  assert isinstance(refs, list)


def test_keras_ghost_mode_discovery():
  from ml_switcheroo.frameworks.base import InitMode

  adapter = KerasAdapter()
  adapter._mode = InitMode.GHOST
  adapter._snapshot_data = {
    "categories": {
      StandardCategory.ACTIVATION.value: [{"name": "test_activation", "api_path": "test", "kind": "function", "args": {}}]
    }
  }
  refs = adapter.collect_api(StandardCategory.ACTIVATION)
  assert len(refs) == 1

  adapter._snapshot_data = None
  refs = adapter.collect_api(StandardCategory.ACTIVATION)
  assert len(refs) == 0


from unittest.mock import patch


def test_keras_scan_edge_cases():
  adapter = KerasAdapter()

  import types

  mock_mod = types.ModuleType("mock_mod")

  # 344: starts with _
  mock_mod._hidden = "hidden"
  # 346: in block_list
  mock_mod.Blocked = "blocked"

  # 351: is_keras_object True
  class MockKerasObj:
    def get_config(self):
      pass

  mock_mod.ValidClass = MockKerasObj

  # Not keras object
  class NotKeras:
    pass

  mock_mod.InvalidClass = NotKeras

  # 357-358: exception in getmembers
  def bug_getmembers(m):
    raise Exception("error")

  with patch("ml_switcheroo.frameworks.keras.inspect.getmembers", side_effect=bug_getmembers):
    assert adapter._scan_module(mock_mod, "prefix", "class") == []

  res = adapter._scan_module(mock_mod, "prefix", "class", block_list={"Blocked"})
  assert len(res) == 1


def test_keras_module_reimport():
  import sys
  import types
  import importlib

  mock_keras = types.ModuleType("keras")
  sys.modules["keras"] = mock_keras
  sys.modules["keras.activations"] = mock_keras
  sys.modules["keras.layers"] = mock_keras
  sys.modules["keras.losses"] = mock_keras
  sys.modules["keras.ops"] = mock_keras
  sys.modules["keras.optimizers"] = mock_keras
  sys.modules["keras.random"] = mock_keras

  import ml_switcheroo.frameworks.keras

  importlib.reload(ml_switcheroo.frameworks.keras)

  # Also test 72 (no snapshot)
  with patch("ml_switcheroo.frameworks.keras.keras", None):
    with patch("ml_switcheroo.frameworks.keras.load_snapshot_for_adapter", return_value=None):
      ml_switcheroo.frameworks.keras.KerasAdapter()

  # Also test 87
  with patch("ml_switcheroo.frameworks.keras.keras", mock_keras):
    adapter = ml_switcheroo.frameworks.keras.KerasAdapter()
    assert adapter.search_modules == ["keras.ops", "keras.layers", "keras.activations", "keras.random"]

  # Also test 247
  with patch("ml_switcheroo.frameworks.keras.load_definitions", return_value={}):
    adapter = ml_switcheroo.frameworks.keras.KerasAdapter()
    assert "ReLU" in adapter.definitions
