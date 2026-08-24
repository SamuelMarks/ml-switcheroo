"""Tests for ml_switcheroo.frameworks.flax_nnx."""

from unittest.mock import MagicMock, patch

from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.frameworks.base import InitMode


def test_flax_nnx_initialization_live():
  """Test live initialization when flax is available."""
  with patch("ml_switcheroo.frameworks.flax_nnx.flax_nnx", MagicMock()):
    adapter = FlaxNNXAdapter()
    assert adapter._mode == InitMode.LIVE
    assert adapter._flax_available is True


def test_flax_nnx_initialization_ghost():
  """Test ghost initialization when flax is not available."""
  ghost_item = {
    "name": "fake",
    "module": "test",
    "tier": SemanticTier.NEURAL.value,
    "api_path": "test.fake",
    "kind": "function",
  }
  with patch("ml_switcheroo.frameworks.flax_nnx.flax_nnx", None):
    with patch(
      "ml_switcheroo.frameworks.flax_nnx.load_snapshot_for_adapter",
      return_value={"categories": {SemanticTier.NEURAL.value: [ghost_item]}},
    ):
      adapter = FlaxNNXAdapter()
      assert adapter._mode == InitMode.GHOST
      assert adapter._flax_available is False
      assert len(adapter._collect_ghost(SemanticTier.NEURAL)) == 1


def test_flax_nnx_initialization_ghost_empty():
  """Test ghost initialization when flax is not available and no snapshot."""
  with patch("ml_switcheroo.frameworks.flax_nnx.flax_nnx", None):
    with patch("ml_switcheroo.frameworks.flax_nnx.load_snapshot_for_adapter", return_value={}):
      adapter = FlaxNNXAdapter()
      assert adapter._mode == InitMode.GHOST
      assert adapter._flax_available is False
      assert len(adapter._collect_ghost(SemanticTier.NEURAL)) == 0


def test_flax_nnx_import_alias():
  """Test import alias properties."""
  adapter = FlaxNNXAdapter()
  assert adapter.import_alias == ("flax.nnx", "nnx")
  assert "flax.nnx" in adapter.import_namespaces


def test_flax_nnx_harness_configs():
  """Test harness configurations."""
  adapter = FlaxNNXAdapter()
  assert "import flax.nnx as nnx" in adapter.test_config["import"]
  assert adapter.harness_imports == ["from flax import nnx"]
  assert "def _make_flax_rngs" in adapter.get_harness_init_code()


def test_flax_nnx_supported_tiers_and_args():
  """Test supported tiers and declared magic args."""
  adapter = FlaxNNXAdapter()
  assert SemanticTier.NEURAL in adapter.supported_tiers
  assert adapter.declared_magic_args == ["rngs"]


def test_flax_nnx_traits():
  """Test structural and plugin traits."""
  adapter = FlaxNNXAdapter()
  st = adapter.structural_traits
  assert st.module_base == "flax.nnx.Module"
  assert st.forward_method == "__call__"
  assert st.requires_super_init is False

  pt = adapter.plugin_traits
  assert pt.requires_explicit_rng is True


def test_flax_nnx_definitions():
  """Test definitions loading."""
  with patch("ml_switcheroo.frameworks.flax_nnx.load_definitions", return_value={"test": MagicMock()}):
    adapter = FlaxNNXAdapter()
    defs = adapter.definitions
    assert "Module" in defs
    assert "ReLU" in defs
    assert "Linear" in defs
    assert "Conv2d" in defs


def test_flax_nnx_convert_with_jax():
  """Test convert when jax.numpy is available."""
  adapter = FlaxNNXAdapter()
  try:
    import jax.numpy as jnp

    res = adapter.convert([1, 2, 3])
    assert jnp.array_equal(res, jnp.array([1, 2, 3]))

    # Exception case
    with patch("jax.numpy.array", side_effect=Exception("error")):
      res2 = adapter.convert([1, 2, 3])
      assert res2 == [1, 2, 3]
  except ImportError:
    pass


def test_flax_nnx_convert_without_jax():
  """Test convert when jax.numpy is not available."""
  adapter = FlaxNNXAdapter()
  # Mocking builtins.__import__
  real_import = __import__

  def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "jax.numpy":
      raise ImportError()
    return real_import(name, globals, locals, fromlist, level)

  with patch("builtins.__import__", side_effect=mock_import):
    res = adapter.convert([1, 2, 3])
    assert res == [1, 2, 3]


def test_flax_nnx_apply_wiring():
  """Test apply_wiring."""
  adapter = FlaxNNXAdapter()
  snapshot = {
    "mappings": {
      "some_op": {"api": "flax.nnx.some_op"},
      "other_op": {"api": "other.op"},
    }
  }
  # Mock _apply_stack_wiring
  adapter._apply_stack_wiring = MagicMock()

  adapter.apply_wiring(snapshot)

  assert snapshot["mappings"]["some_op"]["api"] == "nnx.some_op"
  assert snapshot["mappings"]["other_op"]["api"] == "other.op"
  assert "forward" in snapshot["mappings"]
  assert "register_buffer" in snapshot["mappings"]


def test_flax_nnx_get_tiered_examples():
  """Test tiered examples."""
  adapter = FlaxNNXAdapter()
  examples = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples


def test_flax_nnx_get_doc_url():
  """Test doc URL generation."""
  adapter = FlaxNNXAdapter()
  url = adapter.get_doc_url("flax.nnx.Linear")
  assert "flax.nnx.Linear" in url
