"""Test suite for the Flax Nnx module."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

from ml_switcheroo.frameworks.base import ImportConfig
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter


def test_flax_nnx_adapter_init() -> None:
  """Verify `FlaxNNXAdapter` returns `15` for flax NNX adapter init."""
  adapter = FlaxNNXAdapter()
  assert adapter.display_name == "Flax NNX"
  assert adapter.inherits_from == "jax"
  assert adapter.ui_priority == 15


def test_flax_nnx_import_alias() -> None:
  """Verify `FlaxNNXAdapter` handles flax NNX import alias as expected."""
  adapter = FlaxNNXAdapter()
  assert adapter.import_alias == ("flax.nnx", "nnx")


def test_flax_nnx_import_namespaces() -> None:
  """Verify `FlaxNNXAdapter` emits `flax.nnx` for flax NNX import namespaces."""
  adapter = FlaxNNXAdapter()
  namespaces = adapter.import_namespaces
  assert "flax.nnx" in namespaces
  ns = namespaces["flax.nnx"]
  assert isinstance(ns, ImportConfig)
  assert ns.recommended_alias == "nnx"


def test_flax_nnx_test_config() -> None:
  """Verify `FlaxNNXAdapter` emits `import flax.nnx as nnx` for flax NNX test config."""
  adapter = FlaxNNXAdapter()
  config = adapter.test_config
  assert "import flax.nnx as nnx" in config["import"]
  assert "jax.jit" in config["jit_template"]


def test_flax_nnx_harness_imports() -> None:
  """Verify `FlaxNNXAdapter` emits `from flax import nnx` for flax NNX harness imports."""
  adapter = FlaxNNXAdapter()
  assert "from flax import nnx" in adapter.harness_imports


def test_flax_nnx_harness_init_code() -> None:
  """Verify `FlaxNNXAdapter` emits `def _make_flax_rngs(seed):` for flax NNX harness init code."""
  adapter = FlaxNNXAdapter()
  code = adapter.get_harness_init_code()
  assert "def _make_flax_rngs(seed):" in code
  assert "nnx.Rngs(seed)" in code


def test_flax_nnx_supported_tiers() -> None:
  """Verify `FlaxNNXAdapter` handles flax NNX supported tiers as expected."""
  adapter = FlaxNNXAdapter()
  tiers = adapter.supported_tiers
  assert SemanticTier.ARRAY_API in tiers


def test_flax_nnx_declared_magic_args() -> None:
  """Verify `FlaxNNXAdapter` emits `rngs` for flax NNX declared magic args."""
  adapter = FlaxNNXAdapter()
  assert "rngs" in adapter.declared_magic_args


def test_flax_nnx_structural_traits() -> None:
  """Verify `FlaxNNXAdapter` handles flax NNX structural traits as expected."""
  adapter = FlaxNNXAdapter()
  traits = adapter.structural_traits
  assert traits.module_base == "flax.nnx.Module"
  assert traits.forward_method == "__call__"
  assert not traits.requires_super_init


def test_flax_nnx_definitions() -> None:
  """Verify `FlaxNNXAdapter` emits `Module` for flax NNX definitions."""
  adapter = FlaxNNXAdapter()
  defs = adapter.definitions
  assert "Module" in defs
  assert defs["Module"].api == "flax.nnx.Module"
  assert "relu" in defs
  assert defs["relu"].api == "flax.nnx.relu"


def test_flax_nnx_apply_wiring() -> None:
  """Verify `FlaxNNXAdapter` handles flax NNX apply wiring as expected."""
  adapter = FlaxNNXAdapter()
  snapshot: dict[str, dict[str, JsonValue]] = {"mappings": {}}
  adapter.apply_wiring(snapshot)
  mappings = snapshot["mappings"]
  fwd = mappings["forward"]
  assert isinstance(fwd, dict)
  assert fwd["requires_plugin"] == "inject_training_flag"
  params = mappings["parameters"]
  assert isinstance(params, dict)
  assert params["requires_plugin"] == "torch_parameters_to_nnx"


def test_flax_nnx_apply_wiring_skip() -> None:
  """Verify `FlaxNNXAdapter` handles flax NNX apply wiring skip as expected."""
  adapter = FlaxNNXAdapter()
  snapshot: dict[str, dict[str, JsonValue]] = {"mappings": {"forward": {"api": "already_set"}}}
  adapter.apply_wiring(snapshot)
  fwd = snapshot["mappings"]["forward"]
  assert isinstance(fwd, dict)
  assert fwd["api"] == "already_set"
  assert "requires_plugin" not in fwd


def test_flax_nnx_tiered_examples() -> None:
  """Verify `FlaxNNXAdapter` emits `tier2_neural` for flax NNX tiered examples."""
  adapter = FlaxNNXAdapter()
  examples = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  assert "class Net(nnx.Module):" in examples["tier2_neural"]


def test_flax_nnx_doc_url() -> None:
  """Verify `FlaxNNXAdapter` emits `search.html?q=flax.nnx.relu` for flax NNX doc url."""
  adapter = FlaxNNXAdapter()
  url = adapter.get_doc_url("flax.nnx.relu")
  assert url is not None
  assert "search.html?q=flax.nnx.relu" in url


def test_flax_nnx_convert_logic(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `FlaxNNXAdapter` handles flax NNX convert logic as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  import logging
  import sys
  from unittest.mock import MagicMock

  # test logging warning
  adapter = FlaxNNXAdapter()

  with monkeypatch.context() as m:
    m.setitem(sys.modules, "flax.nnx", None)
    import ml_switcheroo.frameworks.flax_nnx as fn_fw

    m.setattr(fn_fw, "flax_nnx", None)

    # intercept load_snapshot_for_adapter returning None
    m.setattr(fn_fw, "load_snapshot_for_adapter", lambda x: None)

    # Test warning
    with m.context() as m2:
      debug_mock = MagicMock()
      m2.setattr(logging, "debug", debug_mock)
      fn_fw.FlaxNNXAdapter()
      debug_mock.assert_called()

      # also cover live branch
      m2.setattr(fn_fw, "flax_nnx", MagicMock())
      fn_fw.FlaxNNXAdapter()

  # test convert
  import ml_switcheroo.frameworks.flax_nnx as fn_fw

  adapter = fn_fw.FlaxNNXAdapter()
  assert adapter.convert({"a": 1}) == {"a": 1}

  class FakeJNP:
    """Test double `FakeJNP` implementing `array` for flax NNX tests."""

    def array(self, x: list[int]) -> str:
      """Execute `RuntimeError` to support array in flax NNX tests.

      Args:
          x: Input tensor or scalar operand (`list[int]`).

      Returns:
          Resulting `str` from array.
      """
      if x == [1, 2]:
        return "jnp_array"
      raise RuntimeError("fail")

  mock_jax_numpy = FakeJNP()
  mock_jax = MagicMock()
  mock_jax.numpy = mock_jax_numpy
  monkeypatch.setitem(sys.modules, "jax", mock_jax)
  monkeypatch.setitem(sys.modules, "jax.numpy", mock_jax_numpy)
  assert adapter.convert([1, 2]) == "jnp_array"
  assert adapter.convert([3, 4]) == [3, 4]


def test_flax_nnx_definitions_no_mock() -> None:
  """Verify `FlaxNNXAdapter` emits `ReLU` for flax NNX definitions no mock."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  with patch("ml_switcheroo.frameworks.flax_nnx.load_definitions", return_value={}):
    adapter = FlaxNNXAdapter()
    d = adapter.definitions
    assert "ReLU" in d
    assert "Linear" in d
    assert "Conv2d" in d


def test_flax_nnx_convert_branch() -> None:
  """Verify `FlaxNNXAdapter` handles flax NNX convert branch as expected."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()

  class ObjWithArray:
    """An object with __array__."""

    def __array__(self) -> list[int]:
      """Gets the array.

      Returns:
          Resulting `list[int]` from array.
      """
      return [1, 2, 3]

  mock_jnp = MagicMock()
  mock_jnp.array.side_effect = Exception("Fail")
  with patch.dict(sys.modules, {"jax.numpy": mock_jnp}):
    obj = ObjWithArray()
    assert adapter.convert(obj) is obj


def test_flax_nnx_convert_no_import() -> None:
  """Verify `FlaxNNXAdapter` handles flax NNX convert no import as expected."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()

  with patch.dict(sys.modules, {"jax": None, "jax.numpy": None}):
    assert adapter.convert([1, 2, 3]) == [1, 2, 3]


def test_flax_nnx_convert_not_array_like() -> None:
  """Verify `FlaxNNXAdapter` handles flax NNX convert not array like as expected."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()
  mock_jnp = MagicMock()
  with patch.dict(sys.modules, {"jax.numpy": mock_jnp}):
    assert adapter.convert("not_array_like") == "not_array_like"


def test_flax_nnx_apply_wiring_branches() -> None:
  """Verify `FlaxNNXAdapter` handles flax NNX apply wiring branches as expected."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()
  snapshot: dict[str, dict[str, JsonValue]] = {
    "mappings": {
      "test1": {"api": "flax.nnx.Test"},
      "test2": {"other": "flax.nnx.Test"},
      "test3": None,
      "forward": {"api": "fwd"},
      "__call__": {},
      "call": {},
    }
  }
  adapter.apply_wiring(snapshot)
  test1 = snapshot["mappings"]["test1"]
  assert isinstance(test1, dict)
  assert test1["api"] == "nnx.Test"
  dunder_call = snapshot["mappings"]["__call__"]
  assert isinstance(dunder_call, dict)
  assert dunder_call["requires_plugin"] == "inject_training_flag"
  call_op = snapshot["mappings"]["call"]
  assert isinstance(call_op, dict)
  assert call_op["requires_plugin"] == "inject_training_flag"


def test_flax_nnx_reload_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `importlib.reload` handles flax NNX reload exceptions as expected.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  import importlib

  with patch.dict("sys.modules", {"jax": None, "flax.nnx": None}):
    import ml_switcheroo.frameworks.flax_nnx as fnx

    importlib.reload(fnx)
    assert fnx.jax is None
    assert fnx.flax_nnx is None

    # Test __init__ without flax_nnx and missing snapshot
    with patch("ml_switcheroo.frameworks.flax_nnx.load_snapshot_for_adapter", return_value={}):
      adapter = fnx.FlaxNNXAdapter()
      assert adapter._flax_available is False
      assert adapter._mode.name == "GHOST"


def test_flax_nnx_array_exception() -> None:
  """Verify `FlaxNNXAdapter` handles flax NNX array exception as expected."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()

  class FakeArray:
    """A fake array class."""

    def __array__(self) -> list[int]:
      """Gets the array.

      Returns:
          Resulting `list[int]` from array.
      """
      return [1, 2, 3]

  # Actually just patching sys.modules
  mock_jnp = MagicMock()
  mock_jnp.array.side_effect = Exception("Fail")

  with patch.dict(sys.modules, {"jax.numpy": mock_jnp}):
    obj = FakeArray()
    res = adapter.convert(obj)
    assert res is obj


def test_flax_nnx_ghost_mode(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `fnx.FlaxNNXAdapter` returns `1` for flax NNX ghost mode.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  import ml_switcheroo.frameworks.flax_nnx as fnx

  monkeypatch.setattr(fnx, "flax_nnx", None)

  with patch(
    "ml_switcheroo.frameworks.flax_nnx.load_snapshot_for_adapter",
    return_value={
      "categories": {
        "extras": [{"name": "fake", "api_path": "nnx.fake", "kind": "class", "group": "class", "params": []}]
      }
    },
  ):
    from ml_switcheroo.frameworks.base import FrameworkSnapshot

    adapter = fnx.FlaxNNXAdapter()
    assert adapter._mode.name == "GHOST"
    ghosts = adapter._collect_ghost(SemanticTier.EXTRAS)
    assert len(ghosts) == 1
    assert ghosts[0].api_path == "nnx.fake"

    # Test empty snapshot handling
    adapter._snapshot_data = FrameworkSnapshot()
    assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_flax_nnx_properties() -> None:
  """Verify `FlaxNNXAdapter` emits `flax.linen` for flax NNX properties."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()
  assert adapter.import_alias == ("flax.nnx", "nnx")
  assert "flax.linen" in adapter.import_namespaces
  assert adapter.get_harness_init_code()
  assert SemanticTier.NEURAL in adapter.supported_tiers
  assert adapter.declared_magic_args == ["rngs"]
  assert adapter.structural_traits.module_base == "flax.nnx.Module"
  assert adapter.plugin_traits.requires_functional_state is True
  assert "tier4_qwen3-vl" in adapter.get_tiered_examples()
  assert adapter.get_doc_url("flax.nnx.Module") == "https://flax.readthedocs.io/en/latest/search.html?q=flax.nnx.Module"
  assert "flax.nnx as nnx" in adapter.test_config["import"]
  assert "from flax import nnx" in adapter.harness_imports


def test_flax_nnx_definitions_extra(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `fnx.FlaxNNXAdapter` emits `ReLU` for flax NNX definitions extra.

  Args:
      monkeypatch: Pytest monkeypatch fixture for modifying attributes or environment.
  """
  import ml_switcheroo.frameworks.flax_nnx as fnx

  with patch("ml_switcheroo.frameworks.flax_nnx.load_definitions", return_value={}):
    adapter = fnx.FlaxNNXAdapter()
    defs = adapter.definitions
    assert "ReLU" in defs
    assert "Linear" in defs
    assert "Conv2d" in defs
    assert "Module" in defs


def test_flax_nnx_apply_wiring_extra() -> None:
  """Verify `FlaxNNXAdapter` handles flax NNX apply wiring extra as expected."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()
  snapshot: dict[str, dict[str, JsonValue]] = {"mappings": {"test_api": {"api": "flax.nnx.SomeModule"}}}
  adapter.apply_wiring(snapshot)
  test_api = snapshot["mappings"]["test_api"]
  assert isinstance(test_api, dict)
  assert test_api["api"] == "nnx.SomeModule"
  fwd = snapshot["mappings"]["forward"]
  assert isinstance(fwd, dict)
  assert fwd["requires_plugin"] == "inject_training_flag"
  reg_buf = snapshot["mappings"]["register_buffer"]
  assert isinstance(reg_buf, dict)
  assert reg_buf["requires_plugin"] == "torch_register_buffer_to_nnx"


def test_flax_nnx_collect_ghost_no_snapshot() -> None:
  """Verify `FlaxNNXAdapter` handles flax NNX collect ghost no snapshot as expected."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()
  adapter._snapshot_data = None
  assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_flax_nnx_definitions_prepopulated() -> None:
  """Verifies definitions when ReLU, Linear, Conv2d are already present."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter, StandardMap

  custom_defs = {
    "ReLU": StandardMap(api="custom.relu"),
    "Linear": StandardMap(api="custom.linear"),
    "Conv2d": StandardMap(api="custom.conv2d"),
  }
  with patch("ml_switcheroo.frameworks.flax_nnx.load_definitions", return_value=custom_defs):
    adapter = FlaxNNXAdapter()
    defs = adapter.definitions
    assert defs["ReLU"].api == "custom.relu"
    assert defs["Linear"].api == "custom.linear"
    assert defs["Conv2d"].api == "custom.conv2d"
