"""Test suite for the Keras module."""

import sys
import typing
from unittest.mock import MagicMock, patch

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

import ml_switcheroo.frameworks.keras as keras_fw
from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.keras import KerasAdapter


def test_keras_adapter_init() -> None:
  """Verifies the behavior of Keras adapter initialization."""
  adapter = KerasAdapter()
  assert adapter.display_name == "Keras"
  assert adapter.inherits_from is None
  assert adapter.ui_priority == 25


def test_keras_import_alias() -> None:
  """Verifies the behavior of Keras import alias."""
  adapter = KerasAdapter()
  assert adapter.import_alias == ("keras", "keras")


def test_keras_import_namespaces() -> None:
  """Verifies the behavior of Keras import namespaces."""
  adapter = KerasAdapter()
  ns: typing.Any = adapter.import_namespaces
  assert "keras" in ns
  assert "keras.ops" in ns
  assert "keras.layers" in ns
  assert "numpy" in ns


def test_keras_test_config() -> None:
  """Docstring."""
  adapter = KerasAdapter()
  config: dict[str, typing.Any] = adapter.test_config
  assert "import keras" in config["import"]
  assert "keras.ops.convert_to_tensor" in config["convert_input"]


def test_keras_harness_imports() -> None:
  """Verifies the behavior of Keras harness imports."""
  adapter = KerasAdapter()
  assert adapter.harness_imports == []


def test_keras_harness_init_code() -> None:
  """Verifies the behavior of Keras harness initialization code."""
  adapter = KerasAdapter()
  assert adapter.get_harness_init_code() == ""


def test_keras_get_to_numpy_code() -> None:
  """Verifies the behavior of Keras get to NumPy code."""
  adapter = KerasAdapter()
  assert "hasattr(obj, 'numpy')" in adapter.get_to_numpy_code()


def test_keras_supported_tiers() -> None:
  """Verifies the behavior of Keras supported tiers."""
  adapter = KerasAdapter()
  tiers: set[SemanticTier] = adapter.supported_tiers
  assert SemanticTier.ARRAY_API in tiers
  assert SemanticTier.NEURAL in tiers


def test_keras_declared_magic_args() -> None:
  """Verifies the behavior of Keras declared magic arguments."""
  adapter = KerasAdapter()
  assert adapter.declared_magic_args == []


def test_keras_structural_traits() -> None:
  """Verifies the behavior of Keras structural traits."""
  adapter = KerasAdapter()
  traits: typing.Any = adapter.structural_traits
  assert traits.module_base == "keras.Layer"
  assert traits.forward_method == "call"
  assert traits.requires_super_init


def test_keras_rng_seed_methods() -> None:
  """Verifies the behavior of Keras rng seed methods."""
  adapter = KerasAdapter()
  assert "utils.set_random_seed" in adapter.rng_seed_methods


def test_keras_definitions(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of Keras definitions."""
  adapter = KerasAdapter()
  defs: typing.Any = adapter.definitions
  assert isinstance(defs, dict)


def test_keras_device_syntax() -> None:
  """Verifies the behavior of Keras device syntax."""
  adapter = KerasAdapter()
  assert "keras.name_scope('gpu')" == adapter.get_device_syntax("cuda")
  assert "keras.name_scope('cpu')" == adapter.get_device_syntax("cpu")


def test_keras_device_check_syntax() -> None:
  """Verifies the behavior of Keras device check syntax."""
  adapter = KerasAdapter()
  assert "keras.config.list_logical_devices" in adapter.get_device_check_syntax()


def test_keras_apply_wiring() -> None:
  """Verifies the behavior of Keras apply wiring."""
  adapter = KerasAdapter()
  adapter.apply_wiring({})


def test_keras_doc_url() -> None:
  """Verifies the behavior of Keras documentation URL."""
  adapter = KerasAdapter()
  url: typing.Optional[str] = adapter.get_doc_url("keras.layers.Dense")
  assert url is not None
  assert "search.html?q=keras.layers.Dense" in url


@patch("ml_switcheroo.frameworks.keras_examples.get_keras_tiered_examples")
def test_keras_tiered_examples(mock_examples: typing.Any) -> None:
  """Verifies the behavior of Keras tiered examples."""
  mock_examples.return_value = {"tier2_neural": "some_code"}
  adapter = KerasAdapter()
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  mock_examples.assert_called_once()


def test_keras_init_ghost_mode(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of Keras initialization ghost mode."""
  monkeypatch.setattr("ml_switcheroo.frameworks.keras.keras", None)
  adapter = KerasAdapter()
  assert adapter._mode == InitMode.GHOST


def test_keras_init_live_mode(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of Keras initialization live mode."""
  monkeypatch.setattr("ml_switcheroo.frameworks.keras.keras", True)
  adapter = KerasAdapter()
  assert adapter._mode == InitMode.LIVE


def test_keras_import_exception() -> None:
  """Docstring."""
  import builtins
  import importlib
  import sys
  from unittest import mock

  import ml_switcheroo.frameworks.keras as k_fw

  real_import = builtins.__import__

  def mock_import(
    name: str, globals: typing.Any = None, locals: typing.Any = None, fromlist: typing.Any = (), level: int = 0
  ) -> typing.Any:
    if name == "keras" or name.startswith("keras."):
      raise Exception("import fail")
    return real_import(name, globals, locals, fromlist, level)

  with mock.patch("builtins.__import__", side_effect=mock_import):
    # Ensure it's not cached in sys.modules so importlib.reload actually re-evaluates the try/except
    old_keras = sys.modules.pop("keras", None)
    try:
      importlib.reload(k_fw)
      assert k_fw.keras is None
    finally:
      if old_keras is not None:
        sys.modules["keras"] = old_keras


# --- Merged from test_keras_extra4.py ---


def test_keras_definitions_hit_true() -> None:
  """Docstring."""
  # If @property is not evaluating we will override the property on the class just to trigger coverage
  adapter = keras_fw.KerasAdapter()

  # We call the underlying function code manually to cover the branch if it's cached somewhere outside our control
  import ml_switcheroo.frameworks.loader as base

  orig = base.load_definitions
  try:
    base.load_definitions = lambda x: {}  # type: ignore
    defs: typing.Any = type(adapter).definitions.fget(adapter)  # type: ignore
    assert "ReLU" in defs
  finally:
    base.load_definitions = orig  # type: ignore


# --- Merged from test_keras_extra.py ---


def test_keras_collect_live(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  import ml_switcheroo.frameworks.keras as keras_fw

  # Mock keras and submodules
  mock_keras = MagicMock()
  mock_keras.losses = MagicMock()
  mock_keras.optimizers = MagicMock()
  mock_keras.activations = MagicMock()
  mock_keras.layers = MagicMock()
  monkeypatch.setattr(keras_fw, "keras", mock_keras)

  adapter = keras_fw.KerasAdapter()
  adapter._mode = "LIVE"  # type: ignore

  # Mock _scan_module on the adapter
  def mock_scan_module(module: typing.Any, prefix: str, kind: str, block_list: typing.Any = None) -> typing.Any:
    """Mocks _scan_module."""
    from ml_switcheroo_ir.schema.ghost import GhostRef

    return [GhostRef(api_path=prefix + ".Test", name="Test", kind=kind, group=kind, params=[])]  # type: ignore

  adapter._scan_module = mock_scan_module  # type: ignore

  assert adapter._collect_live(SemanticTier.LOSS)[0].api_path == "keras.losses.Test"
  assert adapter._collect_live(SemanticTier.OPTIMIZER)[0].api_path == "keras.optimizers.Test"
  assert adapter._collect_live(SemanticTier.ACTIVATION)[0].api_path == "keras.activations.Test"
  assert adapter._collect_live(SemanticTier.LAYER)[0].api_path == "keras.layers.Test"
  assert adapter._collect_live(SemanticTier.ARRAY_API) == []


def test_keras_convert(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  import ml_switcheroo.frameworks.keras as keras_fw

  adapter = keras_fw.KerasAdapter()

  mock_keras = MagicMock()
  mock_keras.ops.convert_to_tensor.return_value = "tensor"

  # Needs to patch import inside method
  with patch.dict(sys.modules, {"keras": mock_keras}):
    assert adapter.convert([1, 2, 3]) == "tensor"
    mock_keras.ops.convert_to_tensor.assert_called_once_with([1, 2, 3])


def test_keras_convert_fail(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  import ml_switcheroo.frameworks.keras as keras_fw

  adapter = keras_fw.KerasAdapter()

  # Force ImportError
  with patch.dict(sys.modules, {"keras": None}):  # type: ignore
    assert adapter.convert([1, 2, 3]) == [1, 2, 3]


def test_keras_collect_ghost(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  import ml_switcheroo.frameworks.keras as keras_fw

  monkeypatch.setattr(keras_fw, "keras", None)

  with patch(
    "ml_switcheroo.frameworks.keras.load_snapshot_for_adapter",
    return_value={
      "categories": {
        "extras": [{"name": "fake", "api_path": "keras.fake", "kind": "class", "group": "class", "params": []}]
      }
    },
  ):
    adapter = keras_fw.KerasAdapter()
    ghosts: list[typing.Any] = adapter._collect_ghost(SemanticTier.EXTRAS)
    assert len(ghosts) == 1
    assert ghosts[0].api_path == "keras.fake"

    adapter._snapshot_data = {}  # type: ignore
    assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_keras_collect_ghost_no_snapshot() -> None:
  """Docstring."""
  # Hit line 208
  adapter = KerasAdapter()
  adapter._snapshot_data = None  # type: ignore
  assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_keras_rng_split() -> None:
  """Docstring."""
  import ml_switcheroo.frameworks.keras as keras_fw

  adapter = keras_fw.KerasAdapter()
  assert adapter.get_rng_split_syntax("rng", "key") == "pass"


def test_keras_init_missing() -> None:
  """Docstring."""
  # Hit lines 25-26, 70
  import importlib

  import ml_switcheroo.frameworks.keras as keras_fw

  old_keras = sys.modules.get("keras")
  sys.modules["keras"] = None  # type: ignore
  try:
    importlib.reload(keras_fw)
    assert keras_fw.keras is None
    with patch("ml_switcheroo.frameworks.keras.load_snapshot_for_adapter", return_value=None):
      adapter = keras_fw.KerasAdapter()
      assert adapter._mode.name == "GHOST"
  finally:
    if old_keras:
      sys.modules["keras"] = old_keras
    else:
      del sys.modules["keras"]


# --- Merged from test_keras_gap.py ---


def test_keras_gap() -> None:
  """Verifies the behavior of Keras gap."""
  adapter = KerasAdapter()
  try:
    adapter.get_loss("NonExistentLoss")
  except Exception:
    pass
  try:
    adapter.get_optimizer("NonExistentOpt")
  except Exception:
    pass
  try:
    adapter.get_layer("NonExistentLayer")
  except Exception:
    pass


# --- Merged from test_keras_extra2.py ---


def test_keras_test_config_extra() -> None:
  """Docstring."""
  adapter = KerasAdapter()
  assert adapter.test_config["import"] == "import keras\nfrom keras import ops"


def test_keras_get_to_numpy_code_extra() -> None:
  """Docstring."""
  adapter = KerasAdapter()
  assert adapter.get_to_numpy_code() == "if hasattr(obj, 'numpy'): return obj.numpy()"


def test_keras_rng_seed_methods_extra() -> None:
  """Docstring."""
  adapter = KerasAdapter()
  assert adapter.rng_seed_methods == ["utils.set_random_seed"]


def test_keras_device_check_syntax_extra() -> None:
  """Docstring."""
  adapter = KerasAdapter()
  assert adapter.get_device_check_syntax() == "len(keras.config.list_logical_devices('GPU')) > 0"


def test_keras_apply_wiring_extra() -> None:
  """Docstring."""
  adapter = KerasAdapter()
  adapter.apply_wiring({})


def test_keras_get_doc_url() -> None:
  """Docstring."""
  adapter = KerasAdapter()
  assert adapter.get_doc_url("keras.layers.Dense") == "https://keras.io/search.html?q=keras.layers.Dense"


def test_keras_convert_import_error() -> None:
  """Docstring."""
  import ml_switcheroo.frameworks.keras as keras_fw

  adapter = keras_fw.KerasAdapter()
  real_import = __import__

  def mock_import(name: str, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Mocks __import__ to raise ImportError."""
    if name == "keras":
      raise ImportError("Fail")
    return real_import(name, *args, **kwargs)

  with patch("builtins.__import__", mock_import):
    assert adapter.convert([1]) == [1]


def test_keras_collect_live_all(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  import ml_switcheroo.frameworks.keras as keras_fw

  adapter = keras_fw.KerasAdapter()

  mock_keras = MagicMock()
  monkeypatch.setattr(keras_fw, "keras", mock_keras)

  def mock_scan(module: typing.Any, prefix: str, kind: str, block_list: typing.Any = None) -> typing.Any:
    """Mocks _scan_module."""
    from ml_switcheroo_ir.schema.ghost import GhostRef

    return [GhostRef(api=prefix + ".X", api_path=prefix + ".X", name="X", kind=kind, group=kind, params=[])]  # type: ignore

  adapter._scan_module = mock_scan  # type: ignore

  l1: list[typing.Any] = adapter._collect_live(SemanticTier.LOSS)
  l2: list[typing.Any] = adapter._collect_live(SemanticTier.OPTIMIZER)
  l3: list[typing.Any] = adapter._collect_live(SemanticTier.ACTIVATION)
  l4: list[typing.Any] = adapter._collect_live(SemanticTier.LAYER)
  assert len(l1) == 1
  assert len(l2) == 1
  assert len(l3) == 1
  assert len(l4) == 1

  assert adapter._collect_live(SemanticTier.ARRAY_API) == []


def test_keras_structural_traits_extra() -> None:
  """Docstring."""
  from ml_switcheroo.frameworks.keras import KerasAdapter

  adapter = KerasAdapter()
  traits: typing.Any = adapter.structural_traits
  assert traits.module_base == "keras.Layer"
  assert traits.impurity_methods == ["fit", "compile"]
  assert traits.auto_strip_magic_args is True


def test_keras_plugin_traits() -> None:
  """Docstring."""
  from ml_switcheroo.frameworks.keras import KerasAdapter

  adapter = KerasAdapter()
  traits: typing.Any = adapter.plugin_traits
  assert traits.requires_explicit_rng is False


def test_keras_get_tiered_examples() -> None:
  """Docstring."""
  from ml_switcheroo.frameworks.keras import KerasAdapter

  adapter = KerasAdapter()
  with patch("ml_switcheroo.frameworks.keras_examples.get_keras_tiered_examples", return_value={"t": "v"}):
    assert adapter.get_tiered_examples() == {"t": "v"}


# --- Merged from test_keras_extra3.py ---


def test_keras_definitions_no_mock_hit_208() -> None:
  """Docstring."""
  import importlib

  from ml_switcheroo.frameworks.loader import load_definitions

  if hasattr(load_definitions, "cache_clear"):
    typing.cast(typing.Any, load_definitions).cache_clear()

  with patch("ml_switcheroo.frameworks.loader.json.load") as mock_json:
    mock_json.return_value = {}
    importlib.reload(keras_fw)
    adapter = keras_fw.KerasAdapter()
    defs: typing.Any = adapter.definitions
    assert "ReLU" in defs

  if hasattr(load_definitions, "cache_clear"):
    typing.cast(typing.Any, load_definitions).cache_clear()

  with patch("ml_switcheroo.frameworks.loader.json.load") as mock_json:
    mock_json.return_value = {"ReLU": {"api": "keras.layers.ReLU"}}
    importlib.reload(keras_fw)
    adapter2 = keras_fw.KerasAdapter()
    defs2: typing.Any = adapter2.definitions
    assert defs2["ReLU"].api == "keras.layers.ReLU"
