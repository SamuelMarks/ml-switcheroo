"""Tests for Keras framework adapters."""

from typing import Dict
from unittest.mock import MagicMock, patch

from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.keras import KerasAdapter
from ml_switcheroo.frameworks.keras_examples import get_keras_tiered_examples


def test_keras_adapter_ghost_init_empty() -> None:
  """Docstring."""
  with patch("ml_switcheroo.frameworks.keras.keras", None):
    with patch("ml_switcheroo.frameworks.keras.load_snapshot_for_adapter", return_value={}):
      adapter: KerasAdapter = KerasAdapter()
      assert adapter._mode == InitMode.GHOST


def test_keras_adapter_properties() -> None:
  """Docstring."""
  adapter: KerasAdapter = KerasAdapter()
  assert adapter.import_alias == ("keras", "keras")
  assert "keras.layers" in adapter.import_namespaces
  assert "import keras" in adapter.test_config["import"]
  pass
  assert adapter.get_harness_init_code() == ""
  assert "obj.numpy()" in adapter.get_to_numpy_code()
  assert adapter.declared_magic_args == []
  assert "utils.set_random_seed" in adapter.rng_seed_methods

  with patch("ml_switcheroo.frameworks.keras.load_definitions", return_value={"test": MagicMock()}):
    defs: Dict[str, dict[str, str]] = adapter.definitions
    assert isinstance(defs, dict)

  pass

  with patch.dict("sys.modules", {"keras": None}):
    assert adapter.convert([1, 2]) == [1, 2]

  adapter.get_device_syntax("cpu", "0")
  adapter.get_device_syntax("cpu")
  adapter.get_device_check_syntax()
  adapter.get_rng_split_syntax("rng", "key")

  adapter.get_tensor_to_numpy_expr("t")
  assert isinstance(adapter.get_tiered_examples(), dict)

  snapshot: Dict[str, Dict[str, Dict[str, str]]] = {}
  adapter.apply_wiring(snapshot)


def test_keras_io_methods() -> None:
  """Docstring."""
  adapter: KerasAdapter = KerasAdapter()
  assert "import keras" in adapter.get_serialization_imports()
  assert "keras.saving.load_model" in adapter.get_serialization_syntax("load", "path")
  assert "obj.save" in adapter.get_serialization_syntax("save", "path", "obj")
  assert adapter.get_serialization_syntax("other", "path") == ""
  assert "import h5py" in adapter.get_weight_conversion_imports()
  assert "h5py" in adapter.get_weight_load_code("path")
  assert "h5py.File" in adapter.get_weight_save_code("obj", "path")


def test_keras_examples() -> None:
  """Docstring."""
  ex: Dict[str, str] = get_keras_tiered_examples()
  assert isinstance(ex, dict)


def test_keras_missing() -> None:
  """Docstring."""
  adapter: KerasAdapter = KerasAdapter()
  adapter.harness_imports
  adapter.plugin_traits
  adapter._collect_ghost(SemanticTier.NEURAL)
  adapter.get_doc_url("keras.layers.Dense")
  adapter.get_doc_url("unknown")


def test_keras_collect_ghost_data() -> None:
  """Docstring."""
  adapter: KerasAdapter = KerasAdapter()
  adapter._snapshot_data = {
    "categories": {
      "neural": [{"name": "fake", "module": "test", "tier": "neural", "api_path": "fake", "kind": "function"}]
    }
  }
  adapter._collect_ghost(SemanticTier.NEURAL)


def test_keras_collect_live() -> None:
  """Docstring."""
  adapter: KerasAdapter = KerasAdapter()
  with patch("ml_switcheroo.frameworks.keras.keras", MagicMock()):
    adapter._collect_live(SemanticTier.LOSS)
    adapter._collect_live(SemanticTier.OPTIMIZER)
    adapter._collect_live(SemanticTier.ACTIVATION)
    adapter._collect_live(SemanticTier.LAYER)


def test_keras_import_success() -> None:
  """Test module-level import when keras is available."""
  import importlib
  import sys

  mock_keras = MagicMock()
  with patch.dict(
    sys.modules,
    {
      "keras": mock_keras,
      "keras.activations": MagicMock(),
      "keras.layers": MagicMock(),
      "keras.losses": MagicMock(),
      "keras.ops": MagicMock(),
      "keras.optimizers": MagicMock(),
      "keras.random": MagicMock(),
    },
  ):
    import ml_switcheroo.frameworks.keras as mod

    importlib.reload(mod)
  importlib.reload(mod)
