import pytest
import sys
from unittest.mock import patch, MagicMock

# Import necessary modules
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter, flax_nnx
from ml_switcheroo.frameworks.base import StandardCategory, InitMode
from ml_switcheroo.core.ghost import GhostRef


def test_flax_nnx_import_properties():
  adapter = FlaxNNXAdapter()

  assert adapter.import_alias == ("flax.nnx", "nnx")
  assert "flax.nnx" in adapter.import_namespaces
  assert "flax.linen" in adapter.import_namespaces

  heuristics = adapter.discovery_heuristics
  assert "neural" in heuristics
  assert "extras" in heuristics

  conf = adapter.test_config
  assert "import flax.nnx as nnx" in conf["import"]

  tiers = adapter.supported_tiers
  assert len(tiers) == 3

  assert adapter.declared_magic_args == ["rngs"]

  traits = adapter.plugin_traits
  assert traits.has_numpy_compatible_arrays is True

  defs = adapter.definitions
  assert "Module" in defs
  assert "Linear" in defs
  assert "Conv2d" in defs
  assert "relu" in defs

  examples = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples
  assert "tier4_qwen3-vl" in examples

  url = adapter.get_doc_url("flax.nnx.Linear")
  assert url == "https://flax.readthedocs.io/en/latest/search.html?q=flax.nnx.Linear"


@patch("ml_switcheroo.frameworks.flax_nnx.flax_nnx", None)
@patch("ml_switcheroo.frameworks.flax_nnx.load_snapshot_for_adapter")
def test_flax_nnx_ghost_mode_init(mock_load):
  mock_load.return_value = {"categories": {"layer": [{"name": "fake", "api_path": "fake", "kind": "function"}]}}
  adapter = FlaxNNXAdapter()
  assert adapter._mode == InitMode.GHOST
  assert adapter._flax_available is False

  # test collect api in ghost mode
  res = adapter.collect_api(StandardCategory.LAYER)
  assert len(res) == 1
  assert res[0].name == "fake"

  # search modules in ghost mode
  assert adapter.search_modules == []

  # test without snapshot
  mock_load.return_value = {}
  adapter2 = FlaxNNXAdapter()
  assert adapter2.collect_api(StandardCategory.LAYER) == []


@patch("ml_switcheroo.frameworks.flax_nnx.flax_nnx", MagicMock())
def test_flax_nnx_live_mode_scan_layers():
  adapter = FlaxNNXAdapter()
  # Force _flax_available to False manually to cover line 119-120
  adapter._flax_available = False
  assert adapter._scan_nnx_layers() == []

  # Restore
  adapter._flax_available = True

  # Test actual scanning logic error
  with patch("inspect.getmembers", side_effect=Exception("Test Error")):
    res = adapter._scan_nnx_layers()
    assert res == []


def test_flax_nnx_apply_wiring():
  adapter = FlaxNNXAdapter()
  snapshot = {"mappings": {"SomeOp": {"api": "flax.nnx.SomeOp"}, "OtherOp": {"api": "other.Op"}}}
  adapter.apply_wiring(snapshot)
  assert snapshot["mappings"]["SomeOp"]["api"] == "nnx.SomeOp"
  assert "forward" in snapshot["mappings"]
  assert snapshot["mappings"]["forward"]["requires_plugin"] == "inject_training_flag"
  assert snapshot["mappings"]["register_buffer"]["requires_plugin"] == "torch_register_buffer_to_nnx"


def test_flax_nnx_convert():
  adapter = FlaxNNXAdapter()

  # Try with mocked import error for jax.numpy
  orig_import = __import__

  def mock_import(name, *args, **kwargs):
    if name == "jax.numpy":
      raise ImportError()
    return orig_import(name, *args, **kwargs)

  with patch("builtins.__import__", side_effect=mock_import):
    res = adapter.convert([1, 2, 3])
    assert res == [1, 2, 3]


def test_flax_nnx_convert_list_with_jax_numpy_present():
  adapter = FlaxNNXAdapter()

  # Try success
  try:
    import jax.numpy as jnp

    res = adapter.convert([1, 2, 3])
    assert hasattr(res, "__array__")
  except ImportError:
    pass

  # Try error
  with patch("jax.numpy.array", side_effect=Exception("error")) if "jax" in sys.modules else patch("builtins.tuple"):
    # if jax not installed, skip error test or mock __import__
    if "jax" in sys.modules:
      res = adapter.convert([1, 2, 3])
      assert res == [1, 2, 3]


def test_flax_nnx_load_defs():
  import ml_switcheroo.frameworks.flax_nnx

  ml_switcheroo.frameworks.flax_nnx._DEFS_CACHE = None
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter
  import ml_switcheroo.frameworks.loader

  # 308, 312, 317
  with __import__("unittest.mock").mock.patch("ml_switcheroo.frameworks.flax_nnx.load_definitions", return_value={}):
    adapter = FlaxNNXAdapter()
    defs = adapter.definitions
    assert "ReLU" in defs
    assert "Linear" in defs
    assert "Conv2d" in defs


def test_flax_nnx_ghost_mode_properties():
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()

  # 218
  assert "from flax import nnx" in adapter.harness_imports

  # 222
  assert "_make_flax_rngs" in adapter.get_harness_init_code()


def test_flax_nnx_scan_layers_edge_cases():
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()

  # We mock inspect.getmembers to yield a tuple with `_` and an object that throws Exception in issubclass
  class MockObj:
    pass

  class BadObj:
    def __subclasscheck__(self, subclass):
      raise Exception("Bad object")

  def mock_getmembers(mod):
    return [
      ("_hidden", MockObj()),
      ("BadClass", "not_a_class"),
      ("ValidClass", type("MockObj", (object,), {})),
      ("Module", MockObj()),
    ]

  with __import__("unittest.mock").mock.patch("inspect.getmembers", side_effect=mock_getmembers):
    with __import__("unittest.mock").mock.patch("ml_switcheroo.frameworks.flax_nnx.flax_nnx", True):
      # To hit 128-134, we need `GhostInspector` to not throw early if it doesn't enter.
      # actually we don't even need to mock `GhostInspector` if it never calls it (due to exception or `_hidden`)
      adapter._scan_nnx_layers()


def test_flax_nnx_scan_layers_success():
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()

  import sys
  import types

  mock_flax = types.ModuleType("flax")
  mock_nnx = types.ModuleType("nnx")
  mock_flax.nnx = mock_nnx
  sys.modules["flax"] = mock_flax
  sys.modules["flax.nnx"] = mock_nnx

  class MockModule:
    pass

  mock_nnx.Module = MockModule

  class MyLayer(MockModule):
    pass

  def mock_getmembers_succ(mod):
    return [("MyLayer", MyLayer)]

  with __import__("unittest.mock").mock.patch("inspect.getmembers", side_effect=mock_getmembers_succ):
    with __import__("unittest.mock").mock.patch(
      "ml_switcheroo.core.ghost.GhostInspector.inspect", side_effect=Exception("error")
    ):
      # it should hit 132
      adapter._scan_nnx_layers()
