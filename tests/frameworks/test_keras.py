"""Unit tests for `KerasAdapter` (`ml_switcheroo.frameworks.keras`)."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

import ml_switcheroo.frameworks.keras as keras_mod
from ml_switcheroo.core.ghost import GhostRef
from ml_switcheroo.frameworks.base import FrameworkSnapshot, InitMode, StandardMap
from ml_switcheroo.frameworks.keras import KerasAdapter
from ml_switcheroo.frameworks.keras_examples import get_keras_tiered_examples
from ml_switcheroo.frameworks.loader import load_definitions


def test_keras_adapter_metadata_traits_io_and_definitions(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `KerasAdapter` metadata, traits, device/serialization/weight syntax, doc URLs, and definitions.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  adapter = KerasAdapter()
  assert adapter.display_name == "Keras"
  assert adapter.inherits_from is None
  assert adapter.ui_priority == 25
  assert adapter.import_alias == ("keras", "keras")

  ns = adapter.import_namespaces
  assert {"keras", "keras.ops", "keras.layers", "numpy"} <= set(ns.keys())
  assert "import keras" in adapter.test_config["import"]
  assert "keras.ops.convert_to_tensor" in adapter.test_config["convert_input"]
  assert adapter.harness_imports == []
  assert adapter.get_harness_init_code() == ""
  assert "obj.numpy()" in adapter.get_to_numpy_code()
  assert {SemanticTier.ARRAY_API, SemanticTier.NEURAL} <= set(adapter.supported_tiers)
  assert adapter.declared_magic_args == []
  assert adapter.rng_seed_methods == ["utils.set_random_seed"]

  s_traits = adapter.structural_traits
  assert s_traits.module_base == "keras.Layer"
  assert s_traits.forward_method == "call"
  assert s_traits.requires_super_init is True
  assert s_traits.impurity_methods == ["fit", "compile"]
  assert s_traits.auto_strip_magic_args is True
  assert adapter.plugin_traits.requires_explicit_rng is False

  assert adapter.get_device_syntax("cuda") == "keras.name_scope('gpu')"
  assert adapter.get_device_syntax("cpu", "0") == "keras.name_scope('cpu')"
  assert adapter.get_device_check_syntax() == "len(keras.config.list_logical_devices('GPU')) > 0"
  assert adapter.get_rng_split_syntax("rng", "key") == "pass"

  assert "import keras" in adapter.get_serialization_imports()
  assert "keras.saving.load_model" in adapter.get_serialization_syntax("load", "path")
  assert "obj.save" in adapter.get_serialization_syntax("save", "path", "obj")
  assert adapter.get_serialization_syntax("other", "path") == ""
  assert "import h5py" in adapter.get_weight_conversion_imports()
  assert "h5py" in adapter.get_weight_load_code("path")
  assert "h5py.File" in adapter.get_weight_save_code("obj", "path")
  assert len(adapter.get_tensor_to_numpy_expr("t")) > 0

  assert adapter.get_doc_url("keras.layers.Dense") == "https://keras.io/search.html?q=keras.layers.Dense"
  assert isinstance(get_keras_tiered_examples(), dict)
  with patch("ml_switcheroo.frameworks.keras_examples.get_keras_tiered_examples", return_value={"tier2_neural": "code"}):
    assert adapter.get_tiered_examples() == {"tier2_neural": "code"}

  snapshot: dict[str, dict[str, JsonValue]] = {}
  adapter.apply_wiring(snapshot)
  assert snapshot == {}

  # `definitions` fallback vs pre-populated
  load_definitions.cache_clear()
  monkeypatch.setattr(keras_mod, "load_definitions", lambda _fw: {})
  assert "ReLU" in adapter.definitions

  monkeypatch.setattr(keras_mod, "load_definitions", lambda _fw: {"ReLU": StandardMap(api="keras.layers.ReLU")})
  assert adapter.definitions["ReLU"].api == "keras.layers.ReLU"


def test_keras_adapter_modes_collection_and_convert(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify LIVE/GHOST initialization, `_collect_ghost`/`_collect_live`, module reload, and `convert` fallbacks.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr(keras_mod, "keras", True)
  assert KerasAdapter()._mode == InitMode.LIVE

  with (
    patch.object(keras_mod, "keras", None),
    patch.object(keras_mod, "load_snapshot_for_adapter", return_value={}),
  ):
    ghost_empty = KerasAdapter()
    assert ghost_empty._mode == InitMode.GHOST
    ghost_empty._snapshot_data = None
    assert ghost_empty._collect_ghost(SemanticTier.EXTRAS) == []

  with (
    patch.object(keras_mod, "keras", None),
    patch.object(
      keras_mod,
      "load_snapshot_for_adapter",
      return_value={
        "categories": {
          "extras": [{"name": "fake", "api_path": "keras.fake", "kind": "class", "params": []}],
        }
      },
    ),
  ):
    ghost_snap = KerasAdapter()
    ghosts = ghost_snap._collect_ghost(SemanticTier.EXTRAS)
    assert len(ghosts) == 1
    assert ghosts[0].api_path == "keras.fake"
    ghost_snap._snapshot_data = FrameworkSnapshot()
    assert ghost_snap._collect_ghost(SemanticTier.EXTRAS) == []

  # `_collect_live` across LOSS, OPTIMIZER, ACTIVATION, LAYER, and unsupported tier
  live_adapter = KerasAdapter()
  monkeypatch.setattr(keras_mod, "keras", MagicMock())
  assert live_adapter._collect_live(SemanticTier.LOSS) == []

  def _mock_scan(
    _module: MagicMock,
    prefix: str,
    kind: str,
    block_list: set[str] | None = None,
  ) -> list[GhostRef]:
    """Return a single `GhostRef` tagged with `prefix`.

    Args:
        _module: Inspected module mock.
        prefix: Dotted module prefix.
        kind: Object kind string.
        block_list: Optional blocklist.

    Returns:
        Single-element `GhostRef` list.
    """
    _ = block_list
    return [GhostRef(api_path=f"{prefix}.Test", name="Test", kind=kind, params=[])]

  monkeypatch.setattr(live_adapter, "_scan_module", _mock_scan, raising=False)
  assert live_adapter._collect_live(SemanticTier.LOSS)[0].api_path == "keras.losses.Test"
  assert live_adapter._collect_live(SemanticTier.OPTIMIZER)[0].api_path == "keras.optimizers.Test"
  assert live_adapter._collect_live(SemanticTier.ACTIVATION)[0].api_path == "keras.activations.Test"
  assert live_adapter._collect_live(SemanticTier.LAYER)[0].api_path == "keras.layers.Test"
  assert live_adapter._collect_live(SemanticTier.ARRAY_API) == []

  # `convert` with mock `keras` vs `ImportError`
  mock_keras = MagicMock()
  mock_keras.ops.convert_to_tensor.return_value = "keras_tensor"
  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "keras", mock_keras)
    assert live_adapter.convert([1, 2, 3]) == "keras_tensor"
    mock_keras.ops.convert_to_tensor.assert_called_once_with([1, 2, 3])

  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "keras", None)
    assert live_adapter.convert([1, 2, 3]) == [1, 2, 3]
    importlib.reload(keras_mod)
    assert keras_mod.keras is None

  with pytest.MonkeyPatch.context() as mp:
    for sub in (
      "keras",
      "keras.activations",
      "keras.layers",
      "keras.losses",
      "keras.ops",
      "keras.optimizers",
      "keras.random",
    ):
      mp.setitem(sys.modules, sub, MagicMock())
    importlib.reload(keras_mod)
    assert KerasAdapter()._mode == InitMode.LIVE

  importlib.reload(keras_mod)
