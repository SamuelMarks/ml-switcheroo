"""Test module."""

import sys
from unittest.mock import MagicMock, patch
from ml_switcheroo_ir.schema.ghost import SemanticTier


def test_flax_nnx_reload_exceptions(monkeypatch):
  """Test function."""
  import importlib

  # Hide jax
  old_jax = sys.modules.get("jax")
  old_flax = sys.modules.get("flax")
  old_flax_nnx = sys.modules.get("flax.nnx")

  sys.modules["jax"] = None
  sys.modules["flax.nnx"] = None

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


def test_flax_nnx_array_exception():
  """Test function."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()

  class FakeArray:
    """A fake array class."""

    def __array__(self):
      """Gets the array."""
      return [1, 2, 3]

  # Actually just patching sys.modules
  mock_jnp = MagicMock()
  mock_jnp.array.side_effect = Exception("Fail")

  with patch.dict(sys.modules, {"jax.numpy": mock_jnp}):
    obj = FakeArray()
    res = adapter.convert(obj)
    assert res is obj


def test_flax_nnx_ghost_mode(monkeypatch):
  """Test function."""
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
    ghosts = adapter._collect_ghost(SemanticTier.EXTRAS)
    assert len(ghosts) == 1
    assert ghosts[0].api_path == "nnx.fake"

    # Test empty snapshot handling
    adapter._snapshot_data = {}
    assert adapter._collect_ghost(SemanticTier.EXTRAS) == []


def test_flax_nnx_properties():
  """Test function."""
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


def test_flax_nnx_definitions(monkeypatch):
  """Test function."""
  import ml_switcheroo.frameworks.flax_nnx as fnx

  with patch("ml_switcheroo.frameworks.flax_nnx.load_definitions", return_value={}):
    adapter = fnx.FlaxNNXAdapter()
    defs = adapter.definitions
    assert "ReLU" in defs
    assert "Linear" in defs
    assert "Conv2d" in defs
    assert "Module" in defs


def test_flax_nnx_apply_wiring():
  """Test function."""
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()
  snapshot = {"mappings": {"test_api": {"api": "flax.nnx.SomeModule"}}}
  adapter.apply_wiring(snapshot)
  assert snapshot["mappings"]["test_api"]["api"] == "nnx.SomeModule"
  assert snapshot["mappings"]["forward"]["requires_plugin"] == "inject_training_flag"
  assert snapshot["mappings"]["register_buffer"]["requires_plugin"] == "torch_register_buffer_to_nnx"


def test_flax_nnx_collect_ghost_no_snapshot():
  """Test function."""
  # Hit line 82
  from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter

  adapter = FlaxNNXAdapter()
  adapter._snapshot_data = None
  assert adapter._collect_ghost(SemanticTier.EXTRAS) == []
