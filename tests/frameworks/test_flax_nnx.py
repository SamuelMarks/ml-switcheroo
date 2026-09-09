"""Test suite for the Flax Nnx module."""

import sys
import typing
from unittest.mock import MagicMock, patch

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter


def test_flax_nnx_adapter_init() -> None:
  """Verifies the behavior of Flax NNX adapter initialization."""
  adapter = FlaxNNXAdapter()
  assert adapter.display_name == "Flax NNX"
  assert adapter.inherits_from == "jax"
  assert adapter.ui_priority == 15


def test_flax_nnx_import_alias() -> None:
  """Verifies the behavior of Flax NNX import alias."""
  adapter = FlaxNNXAdapter()
  assert adapter.import_alias == ("flax.nnx", "nnx")


def test_flax_nnx_import_namespaces() -> None:
  """Verifies the behavior of Flax NNX import namespaces."""
  adapter = FlaxNNXAdapter()
  namespaces: typing.Any = adapter.import_namespaces
  assert "flax.nnx" in namespaces
  assert namespaces["flax.nnx"].recommended_alias == "nnx"


def test_flax_nnx_test_config() -> None:
  """Docstring."""
  adapter = FlaxNNXAdapter()
  config: dict[str, typing.Any] = adapter.test_config
  assert "import flax.nnx as nnx" in config["import"]
  assert "jax.jit" in config["jit_template"]


def test_flax_nnx_harness_imports() -> None:
  """Verifies the behavior of Flax NNX harness imports."""
  adapter = FlaxNNXAdapter()
  assert "from flax import nnx" in adapter.harness_imports


def test_flax_nnx_harness_init_code() -> None:
  """Verifies the behavior of Flax NNX harness initialization code."""
  adapter = FlaxNNXAdapter()
  code: str = adapter.get_harness_init_code()
  assert "def _make_flax_rngs(seed):" in code
  assert "nnx.Rngs(seed)" in code


def test_flax_nnx_supported_tiers() -> None:
  """Verifies the behavior of Flax NNX supported tiers."""
  adapter = FlaxNNXAdapter()
  tiers: list[SemanticTier] = adapter.supported_tiers
  assert SemanticTier.ARRAY_API in tiers


def test_flax_nnx_declared_magic_args() -> None:
  """Verifies the behavior of Flax NNX declared magic arguments."""
  adapter = FlaxNNXAdapter()
  assert "rngs" in adapter.declared_magic_args


def test_flax_nnx_structural_traits() -> None:
  """Verifies the behavior of Flax NNX structural traits."""
  adapter = FlaxNNXAdapter()
  traits: typing.Any = adapter.structural_traits
  assert traits.module_base == "flax.nnx.Module"
  assert traits.forward_method == "__call__"
  assert not traits.requires_super_init


def test_flax_nnx_definitions() -> None:
  """Verifies the behavior of Flax NNX definitions."""
  adapter = FlaxNNXAdapter()
  defs: typing.Any = adapter.definitions
  assert "Module" in defs
  assert defs["Module"].api == "flax.nnx.Module"
  assert "relu" in defs
  assert defs["relu"].api == "flax.nnx.relu"


def test_flax_nnx_apply_wiring() -> None:
  """Verifies the behavior of Flax NNX apply wiring."""
  adapter = FlaxNNXAdapter()
  snapshot: dict[str, typing.Any] = {"mappings": {}}
  adapter.apply_wiring(snapshot)
  mappings: typing.Any = snapshot["mappings"]
  assert mappings["forward"]["requires_plugin"] == "inject_training_flag"
  assert mappings["parameters"]["requires_plugin"] == "torch_parameters_to_nnx"


def test_flax_nnx_apply_wiring_skip() -> None:
  """Verifies the behavior of Flax NNX apply wiring skip."""
  adapter = FlaxNNXAdapter()
  snapshot: dict[str, typing.Any] = {"mappings": {"forward": {"api": "already_set"}}}
  adapter.apply_wiring(snapshot)
  assert snapshot["mappings"]["forward"]["api"] == "already_set"
  assert "requires_plugin" not in snapshot["mappings"]["forward"]


def test_flax_nnx_tiered_examples() -> None:
  """Verifies the behavior of Flax NNX tiered examples."""
  adapter = FlaxNNXAdapter()
  examples: dict[str, str] = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  assert "class Net(nnx.Module):" in examples["tier2_neural"]


def test_flax_nnx_doc_url() -> None:
  """Verifies the behavior of Flax NNX documentation URL."""
  adapter = FlaxNNXAdapter()
  url: typing.Optional[str] = adapter.get_doc_url("flax.nnx.relu")
  assert url is not None
  assert "search.html?q=flax.nnx.relu" in url


def test_flax_nnx_convert_logic(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  import logging
  import sys
  from unittest.mock import MagicMock

  # test logging warning
  adapter = FlaxNNXAdapter()

  with monkeypatch.context() as m:
    m.setitem(sys.modules, "flax.nnx", None)  # type: ignore
    import ml_switcheroo.frameworks.flax_nnx as fn_fw

    m.setattr(fn_fw, "flax_nnx", None)

    # intercept load_snapshot_for_adapter returning None
    m.setattr(fn_fw, "load_snapshot_for_adapter", lambda x: None)

    # Test warning
    with m.context() as m2:
      m2.setattr(logging, "debug", MagicMock())
      fn_fw.FlaxNNXAdapter()
      typing.cast(MagicMock, logging.debug).assert_called()

      # also cover live branch
      m2.setattr(fn_fw, "flax_nnx", MagicMock())
      fn_fw.FlaxNNXAdapter()

  # test convert
  import ml_switcheroo.frameworks.flax_nnx as fn_fw

  adapter = fn_fw.FlaxNNXAdapter()
  assert adapter.convert({"a": 1}) == {"a": 1}

  class FakeJNP:
    """Docstring."""

    def array(self, x: typing.Any) -> typing.Any:
      """Docstring."""
      if x == [1, 2]:
        return "jnp_array"
      raise Exception("fail")

  mock_jax_numpy = FakeJNP()
  mock_jax = MagicMock()
  mock_jax.numpy = mock_jax_numpy
  monkeypatch.setitem(sys.modules, "jax", mock_jax)
  monkeypatch.setitem(sys.modules, "jax.numpy", mock_jax_numpy)
  assert adapter.convert([1, 2]) == "jnp_array"
  assert adapter.convert([3, 4]) == [3, 4]


# --- Merged from test_flax_nnx_extra2.py ---


def test_flax_nnx_definitions_no_mock() -> None:
  """Docstring."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  with patch("ml_switcheroo.frameworks.flax_nnx.load_definitions", return_value={}):
    adapter = FlaxNNXAdapter()
    d: typing.Any = adapter.definitions
    assert "ReLU" in d
    assert "Linear" in d
    assert "Conv2d" in d


def test_flax_nnx_convert_branch() -> None:
  """Docstring."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()

  class ObjWithArray:
    """An object with __array__."""

    def __array__(self) -> list[int]:
      """Gets the array."""
      return [1, 2, 3]

  mock_jnp = MagicMock()
  mock_jnp.array.side_effect = Exception("Fail")
  with patch.dict(sys.modules, {"jax.numpy": mock_jnp}):
    obj: typing.Any = ObjWithArray()
    assert adapter.convert(obj) is obj


def test_flax_nnx_convert_no_import() -> None:
  """Docstring."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()

  real_import = __import__

  def mock_import(name: str, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Mocks __import__ to raise ImportError for jax.numpy."""
    if name == "jax.numpy":
      raise ImportError("No module named jax.numpy")
    return real_import(name, *args, **kwargs)

  with patch("builtins.__import__", mock_import):
    assert adapter.convert([1, 2, 3]) == [1, 2, 3]


def test_flax_nnx_convert_not_array_like() -> None:
  """Docstring."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()
  mock_jnp = MagicMock()
  with patch.dict(sys.modules, {"jax.numpy": mock_jnp}):
    assert adapter.convert("not_array_like") == "not_array_like"


def test_flax_nnx_apply_wiring_branches() -> None:
  """Docstring."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()
  snapshot: dict[str, typing.Any] = {
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
  assert snapshot["mappings"]["test1"]["api"] == "nnx.Test"
  assert snapshot["mappings"]["__call__"]["requires_plugin"] == "inject_training_flag"
  assert snapshot["mappings"]["call"]["requires_plugin"] == "inject_training_flag"


# --- Merged from test_flax_nnx_extra.py ---


def test_flax_nnx_reload_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
  import importlib

  # Hide jax
  old_jax = sys.modules.get("jax")
  old_flax = sys.modules.get("flax")
  old_flax_nnx = sys.modules.get("flax.nnx")

  sys.modules["jax"] = None  # type: ignore
  sys.modules["flax.nnx"] = None  # type: ignore

  try:
    import ml_switcheroo.frameworks.flax_nnx as fnx

    importlib.reload(fnx)
    assert fnx.jax is None
    assert fnx.flax_nnx is None

    # Test __init__ without flax_nnx and missing snapshot
    with patch("ml_switcheroo.frameworks.flax_nnx.load_snapshot_for_adapter", return_value={}):
      adapter = fnx.FlaxNNXAdapter()
      assert adapter._flax_available is False
      assert adapter._mode.name == "GHOST"

  finally:
    if old_jax:
      sys.modules["jax"] = old_jax
    else:
      del sys.modules["jax"]
    if old_flax:
      sys.modules["flax"] = old_flax
    if old_flax_nnx:
      sys.modules["flax.nnx"] = old_flax_nnx
    else:
      del sys.modules["flax.nnx"]


def test_flax_nnx_array_exception() -> None:
  """Docstring."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()

  class FakeArray:
    """A fake array class."""

    def __array__(self) -> list[int]:
      """Gets the array."""
      return [1, 2, 3]

  # Actually just patching sys.modules
  mock_jnp = MagicMock()
  mock_jnp.array.side_effect = Exception("Fail")

  with patch.dict(sys.modules, {"jax.numpy": mock_jnp}):
    obj: typing.Any = FakeArray()
    res: typing.Any = adapter.convert(obj)
    assert res is obj


def test_flax_nnx_ghost_mode(monkeypatch: pytest.MonkeyPatch) -> None:
  """Docstring."""
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
    adapter = fnx.FlaxNNXAdapter()
    assert adapter._mode.name == "GHOST"
    ghosts: list[typing.Any] = adapter._collect_ghost(SemanticTier.EXTRAS)
    assert len(ghosts) == 1
    assert ghosts[0].api_path == "nnx.fake"

    # Test empty snapshot handling
    adapter._snapshot_data = {}  # type: ignore
    assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_flax_nnx_properties() -> None:
  """Docstring."""
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
  """Docstring."""
  import ml_switcheroo.frameworks.flax_nnx as fnx

  with patch("ml_switcheroo.frameworks.flax_nnx.load_definitions", return_value={}):
    adapter = fnx.FlaxNNXAdapter()
    defs: typing.Any = adapter.definitions
    assert "ReLU" in defs
    assert "Linear" in defs
    assert "Conv2d" in defs
    assert "Module" in defs


def test_flax_nnx_apply_wiring_extra() -> None:
  """Docstring."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()
  snapshot: dict[str, typing.Any] = {"mappings": {"test_api": {"api": "flax.nnx.SomeModule"}}}
  adapter.apply_wiring(snapshot)
  assert snapshot["mappings"]["test_api"]["api"] == "nnx.SomeModule"
  assert snapshot["mappings"]["forward"]["requires_plugin"] == "inject_training_flag"
  assert snapshot["mappings"]["register_buffer"]["requires_plugin"] == "torch_register_buffer_to_nnx"


def test_flax_nnx_collect_ghost_no_snapshot() -> None:
  """Docstring."""
  # Hit line 82
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()
  adapter._snapshot_data = None  # type: ignore
  assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_flax_nnx_definitions_prepopulated() -> None:
  """Verifies definitions when ReLU, Linear, Conv2d are already present."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter, StandardMap

  custom_defs: dict[str, StandardMap] = {
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
