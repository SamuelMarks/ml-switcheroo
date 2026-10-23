"""Test suite for the Keras module."""

from ml_switcheroo.enums import RuntimeTensorOrValue

import sys
import types
import typing
from ml_switcheroo.core.ghost import GhostRef
from pydantic import JsonValue
import pytest
from unittest.mock import MagicMock, patch

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
  ns = adapter.import_namespaces
  assert "keras" in ns
  assert "keras.ops" in ns
  assert "keras.layers" in ns
  assert "numpy" in ns


def test_keras_test_config() -> None:
  """Verify keras test config behavior."""
  adapter = KerasAdapter()
  config = adapter.test_config
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
  tiers = adapter.supported_tiers
  assert SemanticTier.ARRAY_API in tiers
  assert SemanticTier.NEURAL in tiers


def test_keras_declared_magic_args() -> None:
  """Verifies the behavior of Keras declared magic arguments."""
  adapter = KerasAdapter()
  assert adapter.declared_magic_args == []


def test_keras_structural_traits() -> None:
  """Verifies the behavior of Keras structural traits."""
  adapter = KerasAdapter()
  traits = adapter.structural_traits
  assert traits.module_base == "keras.Layer"
  assert traits.forward_method == "call"
  assert traits.requires_super_init


def test_keras_rng_seed_methods() -> None:
  """Verifies the behavior of Keras rng seed methods."""
  adapter = KerasAdapter()
  assert "utils.set_random_seed" in adapter.rng_seed_methods


def test_keras_definitions(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of Keras definitions.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  adapter = KerasAdapter()
  defs = adapter.definitions
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
  snapshot: dict[str, RuntimeTensorOrValue] = {}
  getattr({"v": getattr({"v": adapter}, "__getitem__")("v").apply_wiring}, "__getitem__")("v")(
    getattr({"v": snapshot}, "__getitem__")("v")
  )
  assert snapshot == {}


def test_keras_doc_url() -> None:
  """Verifies the behavior of Keras documentation URL."""
  adapter = KerasAdapter()
  url = adapter.get_doc_url("keras.layers.Dense")
  assert url is not None
  assert "search.html?q=keras.layers.Dense" in url


@patch("ml_switcheroo.frameworks.keras_examples.get_keras_tiered_examples")
def test_keras_tiered_examples(mock_examples: MagicMock) -> None:
  """Verifies the behavior of Keras tiered examples.

  Args:
      mock_examples: The mock examples (MagicMock) for test keras tiered examples.
  """
  mock_examples.return_value = {"tier2_neural": "some_code"}
  adapter = KerasAdapter()
  examples = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  mock_examples.assert_called_once()


def test_keras_init_ghost_mode(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of Keras initialization ghost mode.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  monkeypatch.setattr("ml_switcheroo.frameworks.keras.keras", None)
  adapter = KerasAdapter()
  assert adapter._mode == InitMode.GHOST


def test_keras_init_live_mode(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies the behavior of Keras initialization live mode.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  monkeypatch.setattr("ml_switcheroo.frameworks.keras.keras", True)
  adapter = KerasAdapter()
  assert adapter._mode == InitMode.LIVE


def test_keras_import_exception() -> None:
  """Verify keras import exception behavior."""
  import builtins
  import importlib
  import sys
  from unittest import mock

  import ml_switcheroo.frameworks.keras as k_fw

  real_import = builtins.__import__

  def mock_import(
    name: str,
    globals: typing.Mapping[str, JsonValue] | None = None,
    locals: typing.Mapping[str, JsonValue] | None = None,
    fromlist: typing.Sequence[str] = (),
    level: int = 0,
  ) -> types.ModuleType:
    """Execute mock import helper logic.

    Args:
        name: Identifier or operation name string.
        globals: The globals (typing.Mapping[str, JsonValue] | None) for mock import.
        locals: The locals (typing.Mapping[str, JsonValue] | None) for mock import.
        fromlist: The fromlist (typing.Sequence[str]) for mock import.
        level: The level (int) for mock import.

    Returns:
        Resulting `types.ModuleType` from mock import.
    """
    if name == "keras.ops":
      raise ImportError("import fail")
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
      importlib.reload(k_fw)


def test_keras_definitions_hit_true() -> None:
  """Verify keras definitions hit true behavior."""
  # If @property is not evaluating we will override the property on the class just to trigger coverage
  adapter = keras_fw.KerasAdapter()

  from unittest.mock import patch

  # We call the underlying function code manually to cover the branch if it's cached somewhere outside our control

  with patch("ml_switcheroo.frameworks.loader.load_definitions", return_value={}):
    defs = getattr(
      {
        "v": getattr(
          {
            "v": getattr(
              {
                "v": getattr(
                  {"v": getattr({"v": type}, "__getitem__")("v")(getattr({"v": adapter}, "__getitem__")("v"))},
                  "__getitem__",
                )("v").definitions
              },
              "__getitem__",
            )("v").fget
          },
          "__getitem__",
        )("v")(getattr({"v": adapter}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    assert "ReLU" in defs


def test_keras_collect_live(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify keras collect live behavior.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  import ml_switcheroo.frameworks.keras as keras_fw

  # Mock keras and submodules
  mock_keras = MagicMock()
  mock_keras.losses = MagicMock()
  mock_keras.optimizers = MagicMock()
  mock_keras.activations = MagicMock()
  mock_keras.layers = MagicMock()
  monkeypatch.setattr(keras_fw, "keras", mock_keras)

  adapter = keras_fw.KerasAdapter()
  setattr(getattr({"v": adapter}, "__getitem__")("v"), "_mode", getattr({"v": "LIVE"}, "__getitem__")("v"))

  # Mock _scan_module on the adapter
  def mock_scan_module(
    module: MagicMock | None = None,
    prefix: str = "",
    kind: str = "function",
    block_list: set[str] | None = None,
  ) -> list[GhostRef]:
    """Mocks _scan_module.

    Args:
        module: The module (MagicMock | None) for mock scan module.
        prefix: The prefix (str) for mock scan module.
        kind: The kind (str) for mock scan module.
        block_list: The block list (set[str] | None) for mock scan module.

    Returns:
        Resulting `list[GhostRef]` from mock scan module.
    """
    from ml_switcheroo.core.ghost import GhostRef

    return [GhostRef(api_path=prefix + ".Test", name="Test", kind=kind, params=[])]

  setattr(
    getattr({"v": adapter}, "__getitem__")("v"), "_scan_module", getattr({"v": mock_scan_module}, "__getitem__")("v")
  )

  assert adapter._collect_live(SemanticTier.LOSS)[0].api_path == "keras.losses.Test"
  assert adapter._collect_live(SemanticTier.OPTIMIZER)[0].api_path == "keras.optimizers.Test"
  assert adapter._collect_live(SemanticTier.ACTIVATION)[0].api_path == "keras.activations.Test"
  assert adapter._collect_live(SemanticTier.LAYER)[0].api_path == "keras.layers.Test"
  assert adapter._collect_live(SemanticTier.ARRAY_API) == []


def test_keras_convert(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify keras convert behavior.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  import ml_switcheroo.frameworks.keras as keras_fw

  adapter = keras_fw.KerasAdapter()

  mock_keras = MagicMock()
  mock_keras.ops.convert_to_tensor.return_value = "tensor"

  # Needs to patch import inside method
  with patch.dict(sys.modules, {"keras": mock_keras}):
    assert adapter.convert([1, 2, 3]) == "tensor"
    mock_keras.ops.convert_to_tensor.assert_called_once_with([1, 2, 3])


def test_keras_convert_fail(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify keras convert fail behavior.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  import ml_switcheroo.frameworks.keras as keras_fw

  adapter = keras_fw.KerasAdapter()

  # Force ImportError
  with patch.dict(sys.modules, {"keras": None}):
    assert adapter.convert([1, 2, 3]) == [1, 2, 3]


def test_keras_collect_ghost(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify keras collect ghost behavior.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
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
    ghosts = adapter._collect_ghost(SemanticTier.EXTRAS)
    assert len(ghosts) == 1
    assert ghosts[0].api_path == "keras.fake"

    adapter._snapshot_data = {}
    assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_keras_collect_ghost_no_snapshot() -> None:
  """Verify keras collect ghost no snapshot behavior."""
  # Hit line 208
  adapter = KerasAdapter()
  setattr(getattr({"v": adapter}, "__getitem__")("v"), "_snapshot_data", getattr({"v": None}, "__getitem__")("v"))
  assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_keras_rng_split() -> None:
  """Verify keras rng split behavior."""
  import ml_switcheroo.frameworks.keras as keras_fw

  adapter = keras_fw.KerasAdapter()
  assert adapter.get_rng_split_syntax("rng", "key") == "pass"


def test_keras_init_missing() -> None:
  """Verify keras init missing behavior."""
  # Hit lines 25-26, 70
  import importlib

  import ml_switcheroo.frameworks.keras as keras_fw

  with patch.dict("sys.modules", {"keras": None}):
    importlib.reload(keras_fw)
    assert keras_fw.keras is None
    with patch("ml_switcheroo.frameworks.keras.load_snapshot_for_adapter", return_value=None):
      adapter = keras_fw.KerasAdapter()
      assert adapter._mode.name == "GHOST"


def test_keras_gap() -> None:
  """Verifies the behavior of Keras gap."""
  adapter = KerasAdapter()
  assert not hasattr(adapter, "get_loss")
  assert not hasattr(adapter, "get_optimizer")
  assert not hasattr(adapter, "get_layer")


def test_keras_test_config_extra() -> None:
  """Verify keras test config extra behavior."""
  adapter = KerasAdapter()
  assert adapter.test_config["import"] == "import keras\nfrom keras import ops"


def test_keras_get_to_numpy_code_extra() -> None:
  """Verify keras get to numpy code extra behavior."""
  adapter = KerasAdapter()
  assert adapter.get_to_numpy_code() == "if hasattr(obj, 'numpy'): return obj.numpy()"


def test_keras_rng_seed_methods_extra() -> None:
  """Verify keras rng seed methods extra behavior."""
  adapter = KerasAdapter()
  assert adapter.rng_seed_methods == ["utils.set_random_seed"]


def test_keras_device_check_syntax_extra() -> None:
  """Verify keras device check syntax extra behavior."""
  adapter = KerasAdapter()
  assert adapter.get_device_check_syntax() == "len(keras.config.list_logical_devices('GPU')) > 0"


def test_keras_apply_wiring_extra() -> None:
  """Verify keras apply wiring extra behavior."""
  adapter = KerasAdapter()
  snapshot: dict[str, RuntimeTensorOrValue] = {}
  getattr({"v": getattr({"v": adapter}, "__getitem__")("v").apply_wiring}, "__getitem__")("v")(
    getattr({"v": snapshot}, "__getitem__")("v")
  )
  assert snapshot == {}


def test_keras_get_doc_url() -> None:
  """Verify keras get doc url behavior."""
  adapter = KerasAdapter()
  assert adapter.get_doc_url("keras.layers.Dense") == "https://keras.io/search.html?q=keras.layers.Dense"


def test_keras_convert_import_error() -> None:
  """Verify keras convert import error behavior."""
  import ml_switcheroo.frameworks.keras as keras_fw

  adapter = keras_fw.KerasAdapter()
  real_import = __import__

  def mock_import(name: str, *args: JsonValue, **kwargs: JsonValue) -> types.ModuleType:
    """Mocks __import__ to raise ImportError.

    Args:
        name: Identifier or operation name string.
        args: The args (JsonValue) for mock import.
        kwargs: The kwargs (JsonValue) for mock import.

    Returns:
        Resulting `types.ModuleType` from mock import.
    """
    if name == "keras":
      raise ImportError("Fail")
    _ret_v: types.ModuleType = getattr(
      {
        "v": getattr({"v": real_import}, "__getitem__")("v")(
          getattr({"v": name}, "__getitem__")("v"), *args, **getattr({"v": kwargs}, "__getitem__")("v")
        )
      },
      "__getitem__",
    )("v")
    return _ret_v

  with patch("builtins.__import__", mock_import):
    assert adapter.convert([1]) == [1]


def test_keras_collect_live_all(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify keras collect live all behavior.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  import ml_switcheroo.frameworks.keras as keras_fw

  adapter = keras_fw.KerasAdapter()

  mock_keras = MagicMock()
  monkeypatch.setattr(keras_fw, "keras", mock_keras)

  def mock_scan(
    module: MagicMock,
    prefix: str,
    kind: str,
    block_list: set[str] | None = None,
  ) -> list[GhostRef]:
    """Mocks _scan_module.

    Args:
        module: The module (MagicMock) for mock scan.
        prefix: The prefix (str) for mock scan.
        kind: The kind (str) for mock scan.
        block_list: The block list (set[str] | None) for mock scan.

    Returns:
        Resulting `list[GhostRef]` from mock scan.
    """
    from ml_switcheroo.core.ghost import GhostRef

    return [GhostRef(api_path=prefix + ".X", name="X", kind=kind, params=[])]

  setattr(getattr({"v": adapter}, "__getitem__")("v"), "_scan_module", getattr({"v": mock_scan}, "__getitem__")("v"))

  l1 = adapter._collect_live(SemanticTier.LOSS)
  l2 = adapter._collect_live(SemanticTier.OPTIMIZER)
  l3 = adapter._collect_live(SemanticTier.ACTIVATION)
  l4 = adapter._collect_live(SemanticTier.LAYER)
  assert len(l1) == 1
  assert len(l2) == 1
  assert len(l3) == 1
  assert len(l4) == 1

  assert adapter._collect_live(SemanticTier.ARRAY_API) == []


def test_keras_structural_traits_extra() -> None:
  """Verify keras structural traits extra behavior."""
  from ml_switcheroo.frameworks.keras import KerasAdapter

  adapter = KerasAdapter()
  traits = adapter.structural_traits
  assert traits.module_base == "keras.Layer"
  assert traits.impurity_methods == ["fit", "compile"]
  assert traits.auto_strip_magic_args is True


def test_keras_plugin_traits() -> None:
  """Verify keras plugin traits behavior."""
  from ml_switcheroo.frameworks.keras import KerasAdapter

  adapter = KerasAdapter()
  traits = adapter.plugin_traits
  assert traits.requires_explicit_rng is False


def test_keras_get_tiered_examples() -> None:
  """Verify keras get tiered examples behavior."""
  from ml_switcheroo.frameworks.keras import KerasAdapter

  adapter = KerasAdapter()
  with patch("ml_switcheroo.frameworks.keras_examples.get_keras_tiered_examples", return_value={"t": "v"}):
    assert adapter.get_tiered_examples() == {"t": "v"}


def test_keras_definitions_no_mock_hit_208() -> None:
  """Verify keras definitions no mock hit 208 behavior."""
  import importlib

  from ml_switcheroo.frameworks.loader import load_definitions

  if hasattr(load_definitions, "cache_clear"):
    getattr(load_definitions, "cache_clear")()

  with patch("ml_switcheroo.frameworks.loader.json.load") as mock_json:
    mock_json.return_value = {}
    importlib.reload(keras_fw)
    adapter = keras_fw.KerasAdapter()
    defs = adapter.definitions
    assert "ReLU" in defs

  if hasattr(load_definitions, "cache_clear"):
    getattr(load_definitions, "cache_clear")()

  with patch("ml_switcheroo.frameworks.loader.json.load") as mock_json:
    mock_json.return_value = {"ReLU": {"api": "keras.layers.ReLU"}}
    importlib.reload(keras_fw)
    adapter2 = keras_fw.KerasAdapter()
    defs2 = adapter2.definitions
    assert defs2["ReLU"].api == "keras.layers.ReLU"
