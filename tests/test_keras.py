"""Tests for Keras framework adapters."""

import typing
from unittest.mock import MagicMock, patch

from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.keras import KerasAdapter
from ml_switcheroo.frameworks.keras_examples import get_keras_tiered_examples


def test_keras_adapter_ghost_init_empty() -> None:
  """Verify keras adapter ghost init empty behavior."""
  with patch("ml_switcheroo.frameworks.keras.keras", None):
    with patch("ml_switcheroo.frameworks.keras.load_snapshot_for_adapter", return_value={}):
      adapter: KerasAdapter = KerasAdapter()
      assert adapter._mode == InitMode.GHOST


def test_keras_adapter_properties() -> None:
  """Verify keras adapter properties behavior."""
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
    defs = adapter.definitions
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

  snapshot: typing.Dict[str, typing.Dict[str, typing.Dict[str, str]]] = {}
  adapter.apply_wiring(snapshot)  # type: ignore[arg-type]


def test_keras_io_methods() -> None:
  """Verify keras io methods behavior."""
  adapter: KerasAdapter = KerasAdapter()
  assert "import keras" in adapter.get_serialization_imports()
  assert "keras.saving.load_model" in adapter.get_serialization_syntax("load", "path")
  assert "obj.save" in adapter.get_serialization_syntax("save", "path", "obj")
  assert adapter.get_serialization_syntax("other", "path") == ""
  assert "import h5py" in adapter.get_weight_conversion_imports()
  assert "h5py" in adapter.get_weight_load_code("path")
  assert "h5py.File" in adapter.get_weight_save_code("obj", "path")


def test_keras_examples() -> None:
  """Verify keras examples behavior."""
  ex: typing.Dict[str, str] = get_keras_tiered_examples()
  assert isinstance(ex, dict)


def test_keras_missing() -> None:
  """Verify keras missing behavior."""
  adapter: KerasAdapter = KerasAdapter()
  assert isinstance(adapter.harness_imports, list)
  assert adapter.plugin_traits is not None
  adapter._collect_ghost(SemanticTier.NEURAL)
  assert "keras.io" in (adapter.get_doc_url("keras.layers.Dense") or "")
  _res = adapter.get_doc_url("unknown")
  assert _res == "https://keras.io/search.html?q=unknown"


def test_keras_collect_ghost_data() -> None:
  """Verify keras collect ghost data behavior."""
  adapter: KerasAdapter = KerasAdapter()
  adapter._snapshot_data = {
    "categories": {
      "neural": [{"name": "fake", "module": "test", "tier": "neural", "api_path": "fake", "kind": "function"}]
    }
  }
  _res = adapter._collect_ghost(SemanticTier.NEURAL)
  assert len(_res) == 1
  assert _res[0].name == "fake"


def test_keras_collect_live() -> None:
  """Verify keras collect live behavior."""
  adapter: KerasAdapter = KerasAdapter()
  with patch("ml_switcheroo.frameworks.keras.keras", MagicMock()):
    res_loss = adapter._collect_live(SemanticTier.LOSS)
    res_opt = adapter._collect_live(SemanticTier.OPTIMIZER)
    res_act = adapter._collect_live(SemanticTier.ACTIVATION)
    res_layer = adapter._collect_live(SemanticTier.LAYER)
  assert isinstance(res_loss, list)
  assert isinstance(res_opt, list)
  assert isinstance(res_act, list)
  assert isinstance(res_layer, list)


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
  _res = importlib.reload(mod)
  assert hasattr(_res, "KerasAdapter")
