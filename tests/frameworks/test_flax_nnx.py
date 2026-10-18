"""Test suite for the Flax Nnx module."""

import sys
import typing
from unittest.mock import MagicMock, patch

from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter


def test_flax_nnx_adapter_init() -> None:
  """Verifies the behavior of Flax NNX adapter initialization."""
  adapter: typing.Any = FlaxNNXAdapter()
  assert adapter.display_name == "Flax NNX"
  assert adapter.inherits_from == "jax"
  assert adapter.ui_priority == 15


def test_flax_nnx_import_alias() -> None:
  """Verifies the behavior of Flax NNX import alias."""
  adapter: typing.Any = FlaxNNXAdapter()
  assert adapter.import_alias == ("flax.nnx", "nnx")


def test_flax_nnx_import_namespaces() -> None:
  """Verifies the behavior of Flax NNX import namespaces."""
  adapter: typing.Any = FlaxNNXAdapter()
  namespaces: typing.Any = adapter.import_namespaces
  assert "flax.nnx" in namespaces
  assert namespaces["flax.nnx"].recommended_alias == "nnx"


def test_flax_nnx_test_config() -> None:
  """Verify flax nnx test config behavior."""
  adapter: typing.Any = FlaxNNXAdapter()
  config: typing.Any = adapter.test_config
  assert "import flax.nnx as nnx" in config["import"]
  assert "jax.jit" in config["jit_template"]


def test_flax_nnx_harness_imports() -> None:
  """Verifies the behavior of Flax NNX harness imports."""
  adapter: typing.Any = FlaxNNXAdapter()
  assert "from flax import nnx" in adapter.harness_imports


def test_flax_nnx_harness_init_code() -> None:
  """Verifies the behavior of Flax NNX harness initialization code."""
  adapter: typing.Any = FlaxNNXAdapter()
  code: typing.Any = adapter.get_harness_init_code()
  assert "def _make_flax_rngs(seed):" in code
  assert "nnx.Rngs(seed)" in code


def test_flax_nnx_supported_tiers() -> None:
  """Verifies the behavior of Flax NNX supported tiers."""
  adapter: typing.Any = FlaxNNXAdapter()
  tiers: typing.Any = adapter.supported_tiers
  assert SemanticTier.ARRAY_API in tiers


def test_flax_nnx_declared_magic_args() -> None:
  """Verifies the behavior of Flax NNX declared magic arguments."""
  adapter: typing.Any = FlaxNNXAdapter()
  assert "rngs" in adapter.declared_magic_args


def test_flax_nnx_structural_traits() -> None:
  """Verifies the behavior of Flax NNX structural traits."""
  adapter: typing.Any = FlaxNNXAdapter()
  traits: typing.Any = adapter.structural_traits
  assert traits.module_base == "flax.nnx.Module"
  assert traits.forward_method == "__call__"
  assert not traits.requires_super_init


def test_flax_nnx_definitions() -> None:
  """Verifies the behavior of Flax NNX definitions."""
  adapter: typing.Any = FlaxNNXAdapter()
  defs: typing.Any = adapter.definitions
  assert "Module" in defs
  assert defs["Module"].api == "flax.nnx.Module"
  assert "relu" in defs
  assert defs["relu"].api == "flax.nnx.relu"


def test_flax_nnx_apply_wiring() -> None:
  """Verifies the behavior of Flax NNX apply wiring."""
  adapter: typing.Any = FlaxNNXAdapter()
  snapshot: typing.Any = {"mappings": {}}
  adapter.apply_wiring(snapshot)
  mappings: typing.Any = snapshot["mappings"]
  assert mappings["forward"]["requires_plugin"] == "inject_training_flag"
  assert mappings["parameters"]["requires_plugin"] == "torch_parameters_to_nnx"


def test_flax_nnx_apply_wiring_skip() -> None:
  """Verifies the behavior of Flax NNX apply wiring skip."""
  adapter: typing.Any = FlaxNNXAdapter()
  snapshot: typing.Any = {"mappings": {"forward": {"api": "already_set"}}}
  adapter.apply_wiring(snapshot)
  assert snapshot["mappings"]["forward"]["api"] == "already_set"
  assert "requires_plugin" not in snapshot["mappings"]["forward"]


def test_flax_nnx_tiered_examples() -> None:
  """Verifies the behavior of Flax NNX tiered examples."""
  adapter: typing.Any = FlaxNNXAdapter()
  examples: typing.Any = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  assert "class Net(nnx.Module):" in examples["tier2_neural"]


def test_flax_nnx_doc_url() -> None:
  """Verifies the behavior of Flax NNX documentation URL."""
  adapter: typing.Any = FlaxNNXAdapter()
  url: typing.Any = adapter.get_doc_url("flax.nnx.relu")
  assert url is not None
  assert "search.html?q=flax.nnx.relu" in url


def test_flax_nnx_convert_logic(monkeypatch: typing.Any) -> None:
  """Verify flax nnx convert logic behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import logging
  import sys
  from unittest.mock import MagicMock

  # test logging warning
  adapter: typing.Any = FlaxNNXAdapter()

  with monkeypatch.context() as m:
    m.setitem(sys.modules, "flax.nnx", None)
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
    """Test helper class FakeJNP."""

    def array(self, x: typing.Any) -> typing.Any:
      """Execute array helper logic.

      Args:
          x: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      if x == [1, 2]:
        return "jnp_array"
      raise RuntimeError("fail")

  mock_jax_numpy: typing.Any = FakeJNP()
  mock_jax: typing.Any = MagicMock()
  mock_jax.numpy = mock_jax_numpy
  monkeypatch.setitem(sys.modules, "jax", mock_jax)
  monkeypatch.setitem(sys.modules, "jax.numpy", mock_jax_numpy)
  assert adapter.convert([1, 2]) == "jnp_array"
  assert adapter.convert([3, 4]) == [3, 4]


# --- Merged from test_flax_nnx_extra2.py ---


def test_flax_nnx_definitions_no_mock() -> None:
  """Verify flax nnx definitions no mock behavior."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  with patch("ml_switcheroo.frameworks.flax_nnx.load_definitions", return_value={}):
    adapter: typing.Any = FlaxNNXAdapter()
    d: typing.Any = adapter.definitions
    assert "ReLU" in d
    assert "Linear" in d
    assert "Conv2d" in d


def test_flax_nnx_convert_branch() -> None:
  """Verify flax nnx convert branch behavior."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter: typing.Any = FlaxNNXAdapter()

  class ObjWithArray:
    """An object with __array__."""

    def __array__(self) -> typing.Any:
      """Gets the array.

      Returns:
          Computed test result or fixture value.
      """
      return [1, 2, 3]

  mock_jnp: typing.Any = MagicMock()
  mock_jnp.array.side_effect = Exception("Fail")
  with patch.dict(sys.modules, {"jax.numpy": mock_jnp}):
    obj: typing.Any = ObjWithArray()
    assert adapter.convert(obj) is obj


def test_flax_nnx_convert_no_import() -> None:
  """Verify flax nnx convert no import behavior."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter: typing.Any = FlaxNNXAdapter()

  real_import: typing.Any = __import__

  def mock_import(name: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    """Mocks __import__ to raise ImportError for jax.numpy.

    Args:
        name: Test argument or fixture.
        args: Test argument or fixture.
        kwargs: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if name == "jax.numpy":
      raise ImportError("No module named jax.numpy")
    return real_import(name, *args, **kwargs)

  with patch("builtins.__import__", mock_import):
    assert adapter.convert([1, 2, 3]) == [1, 2, 3]


def test_flax_nnx_convert_not_array_like() -> None:
  """Verify flax nnx convert not array like behavior."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter: typing.Any = FlaxNNXAdapter()
  mock_jnp: typing.Any = MagicMock()
  with patch.dict(sys.modules, {"jax.numpy": mock_jnp}):
    assert adapter.convert("not_array_like") == "not_array_like"


def test_flax_nnx_apply_wiring_branches() -> None:
  """Verify flax nnx apply wiring branches behavior."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter: typing.Any = FlaxNNXAdapter()
  snapshot: typing.Any = {
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


def test_flax_nnx_reload_exceptions(monkeypatch: typing.Any) -> None:
  """Verify flax nnx reload exceptions behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import importlib

  with patch.dict("sys.modules", {"jax": None, "flax.nnx": None}):
    import ml_switcheroo.frameworks.flax_nnx as fnx

    importlib.reload(fnx)
    assert fnx.jax is None
    assert fnx.flax_nnx is None

    # Test __init__ without flax_nnx and missing snapshot
    with patch("ml_switcheroo.frameworks.flax_nnx.load_snapshot_for_adapter", return_value={}):
      adapter: typing.Any = fnx.FlaxNNXAdapter()
      assert adapter._flax_available is False
      assert adapter._mode.name == "GHOST"


def test_flax_nnx_array_exception() -> None:
  """Verify flax nnx array exception behavior."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter: typing.Any = FlaxNNXAdapter()

  class FakeArray:
    """A fake array class."""

    def __array__(self) -> typing.Any:
      """Gets the array.

      Returns:
          Computed test result or fixture value.
      """
      return [1, 2, 3]

  # Actually just patching sys.modules
  mock_jnp: typing.Any = MagicMock()
  mock_jnp.array.side_effect = Exception("Fail")

  with patch.dict(sys.modules, {"jax.numpy": mock_jnp}):
    obj: typing.Any = FakeArray()
    res: typing.Any = adapter.convert(obj)
    assert res is obj


def test_flax_nnx_ghost_mode(monkeypatch: typing.Any) -> None:
  """Verify flax nnx ghost mode behavior.

  Args:
      monkeypatch: Test argument or fixture.
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
    adapter: typing.Any = fnx.FlaxNNXAdapter()
    assert adapter._mode.name == "GHOST"
    ghosts: typing.Any = adapter._collect_ghost(SemanticTier.EXTRAS)
    assert len(ghosts) == 1
    assert ghosts[0].api_path == "nnx.fake"

    # Test empty snapshot handling
    adapter._snapshot_data = {}
    assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_flax_nnx_properties() -> None:
  """Verify flax nnx properties behavior."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter: typing.Any = FlaxNNXAdapter()
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


def test_flax_nnx_definitions_extra(monkeypatch: typing.Any) -> None:
  """Verify flax nnx definitions extra behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
  import ml_switcheroo.frameworks.flax_nnx as fnx

  with patch("ml_switcheroo.frameworks.flax_nnx.load_definitions", return_value={}):
    adapter: typing.Any = fnx.FlaxNNXAdapter()
    defs: typing.Any = adapter.definitions
    assert "ReLU" in defs
    assert "Linear" in defs
    assert "Conv2d" in defs
    assert "Module" in defs


def test_flax_nnx_apply_wiring_extra() -> None:
  """Verify flax nnx apply wiring extra behavior."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter: typing.Any = FlaxNNXAdapter()
  snapshot: typing.Any = {"mappings": {"test_api": {"api": "flax.nnx.SomeModule"}}}
  adapter.apply_wiring(snapshot)
  assert snapshot["mappings"]["test_api"]["api"] == "nnx.SomeModule"
  assert snapshot["mappings"]["forward"]["requires_plugin"] == "inject_training_flag"
  assert snapshot["mappings"]["register_buffer"]["requires_plugin"] == "torch_register_buffer_to_nnx"


def test_flax_nnx_collect_ghost_no_snapshot() -> None:
  """Verify flax nnx collect ghost no snapshot behavior."""
  # Hit line 82
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter: typing.Any = FlaxNNXAdapter()
  adapter._snapshot_data = None
  assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_flax_nnx_definitions_prepopulated() -> None:
  """Verifies definitions when ReLU, Linear, Conv2d are already present."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter, StandardMap

  custom_defs: typing.Any = {
    "ReLU": StandardMap(api="custom.relu"),
    "Linear": StandardMap(api="custom.linear"),
    "Conv2d": StandardMap(api="custom.conv2d"),
  }
  with patch("ml_switcheroo.frameworks.flax_nnx.load_definitions", return_value=custom_defs):
    adapter: typing.Any = FlaxNNXAdapter()
    defs: typing.Any = adapter.definitions
    assert defs["ReLU"].api == "custom.relu"
    assert defs["Linear"].api == "custom.linear"
    assert defs["Conv2d"].api == "custom.conv2d"
