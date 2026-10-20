"""Tests for ml_switcheroo.frameworks.flax_nnx."""

import typing
from unittest.mock import MagicMock, patch

from ml_switcheroo_ir.schema.ghost import SemanticTier

from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.flax_nnx import FlaxNNXAdapter


def test_flax_nnx_initialization_live() -> None:
  """Verify flax nnx initialization live behavior."""
  with patch("ml_switcheroo.frameworks.flax_nnx.flax_nnx", MagicMock()):
    adapter: FlaxNNXAdapter = FlaxNNXAdapter()
    assert adapter._mode == InitMode.LIVE
    assert adapter._flax_available is True


def test_flax_nnx_initialization_ghost() -> None:
  """Verify flax nnx initialization ghost behavior."""
  ghost_item: typing.Dict[str, str] = {
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
      adapter: FlaxNNXAdapter = FlaxNNXAdapter()
      assert adapter._mode == InitMode.GHOST
      assert adapter._flax_available is False
      assert len(adapter._collect_ghost(SemanticTier.NEURAL)) == 1


def test_flax_nnx_initialization_ghost_empty() -> None:
  """Verify flax nnx initialization ghost empty behavior."""
  with patch("ml_switcheroo.frameworks.flax_nnx.flax_nnx", None):
    with patch("ml_switcheroo.frameworks.flax_nnx.load_snapshot_for_adapter", return_value={}):
      adapter: FlaxNNXAdapter = FlaxNNXAdapter()
      assert adapter._mode == InitMode.GHOST
      assert adapter._flax_available is False
      assert len(adapter._collect_ghost(SemanticTier.NEURAL)) == 0


def test_flax_nnx_import_alias() -> None:
  """Verify flax nnx import alias behavior."""
  adapter: FlaxNNXAdapter = FlaxNNXAdapter()
  assert adapter.import_alias == ("flax.nnx", "nnx")
  assert "flax.nnx" in adapter.import_namespaces


def test_flax_nnx_harness_configs() -> None:
  """Verify flax nnx harness configs behavior."""
  adapter: FlaxNNXAdapter = FlaxNNXAdapter()
  assert "import flax.nnx as nnx" in adapter.test_config["import"]
  assert adapter.harness_imports == ["from flax import nnx"]
  assert "def _make_flax_rngs" in adapter.get_harness_init_code()


def test_flax_nnx_supported_tiers_and_args() -> None:
  """Verify flax nnx supported tiers and args behavior."""
  adapter: FlaxNNXAdapter = FlaxNNXAdapter()
  assert SemanticTier.NEURAL in adapter.supported_tiers
  assert adapter.declared_magic_args == ["rngs"]


def test_flax_nnx_traits() -> None:
  """Verify flax nnx traits behavior."""
  adapter: FlaxNNXAdapter = FlaxNNXAdapter()
  st = adapter.structural_traits
  assert st.module_base == "flax.nnx.Module"
  assert st.forward_method == "__call__"
  assert st.requires_super_init is False

  pt = adapter.plugin_traits
  assert pt.requires_explicit_rng is True


def test_flax_nnx_definitions() -> None:
  """Verify flax nnx definitions behavior."""
  with patch("ml_switcheroo.frameworks.flax_nnx.load_definitions", return_value={"test": MagicMock()}):
    adapter: FlaxNNXAdapter = FlaxNNXAdapter()
    defs = adapter.definitions
    assert "Module" in defs
    assert "ReLU" in defs
    assert "Linear" in defs
    assert "Conv2d" in defs


def test_flax_nnx_convert_with_jax() -> None:
  """Verify flax nnx convert with jax behavior."""
  adapter: FlaxNNXAdapter = FlaxNNXAdapter()
  try:
    import jax.numpy as jnp

    res = adapter.convert([1, 2, 3])
    assert jnp.array_equal(res, jnp.array([1, 2, 3]))

    # Exception case
    with patch("jax.numpy.array", side_effect=ValueError("error")):
      res2 = adapter.convert([1, 2, 3])
      assert res2 == [1, 2, 3]
  except ImportError:
    pass


def test_flax_nnx_convert_without_jax() -> None:
  """Verify flax nnx convert without jax behavior."""
  adapter: FlaxNNXAdapter = FlaxNNXAdapter()
  # Mocking builtins.__import__
  import builtins

  real_import = builtins.__import__

  import types

  def mock_import(
    name: str,
    globals: typing.Optional[typing.Dict[str, str]] = None,
    locals: typing.Optional[typing.Dict[str, str]] = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
  ) -> types.ModuleType:
    """Execute mock import helper logic.

    Args:
        name: Test argument or fixture.
        globals: Test argument or fixture.
        locals: Test argument or fixture.
        fromlist: Test argument or fixture.
        level: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if name == "jax.numpy":
      raise ImportError()
    return real_import(name, globals, locals, fromlist, level)

  with patch("builtins.__import__", side_effect=mock_import):
    res = adapter.convert([1, 2, 3])
    assert res == [1, 2, 3]

  # When jax is present: test non-array (243->248), success array, and exception in jnp.array
  mock_jax = MagicMock()
  mock_jnp = MagicMock()
  mock_jax.numpy = mock_jnp
  with patch.dict("sys.modules", {"jax": mock_jax, "jax.numpy": mock_jnp}):
    # Non-array data jumps to line 248
    assert adapter.convert(123) == 123
    # Valid array data
    mock_jnp.array.return_value = "jax_array"
    assert adapter.convert([1, 2]) == "jax_array"
    # Exception in jnp.array
    mock_jnp.array.side_effect = RuntimeError("err")
    assert adapter.convert([1, 2]) == [1, 2]


def test_flax_nnx_import_success() -> None:
  """Test module-level import when flax.nnx is available."""
  import importlib
  import sys

  mock_flax = MagicMock()
  mock_nnx = MagicMock()
  mock_flax.nnx = mock_nnx
  mock_jax = MagicMock()
  with patch.dict(sys.modules, {"flax": mock_flax, "flax.nnx": mock_nnx, "jax": mock_jax}):
    import ml_switcheroo.frameworks.flax_nnx as mod

    importlib.reload(mod)
  _res = importlib.reload(mod)
  assert hasattr(_res, "FlaxNNXAdapter")


def test_flax_nnx_apply_wiring() -> None:
  """Verify flax nnx apply wiring behavior."""
  adapter: FlaxNNXAdapter = FlaxNNXAdapter()
  snapshot: typing.Dict[str, typing.Dict[str, typing.Dict[str, str]]] = {
    "mappings": {
      "some_op": {"api": "flax.nnx.some_op"},
      "other_op": {"api": "other.op"},
    }
  }
  # Mock _apply_stack_wiring
  setattr(adapter, "_apply_stack_wiring", MagicMock())

  adapter.apply_wiring(snapshot)  # type: ignore[arg-type]

  assert snapshot["mappings"]["some_op"]["api"] == "nnx.some_op"
  assert snapshot["mappings"]["other_op"]["api"] == "other.op"
  assert "forward" in snapshot["mappings"]
  assert "register_buffer" in snapshot["mappings"]


def test_flax_nnx_get_tiered_examples() -> None:
  """Verify flax nnx get tiered examples behavior."""
  adapter: FlaxNNXAdapter = FlaxNNXAdapter()
  examples = adapter.get_tiered_examples()
  assert "tier2_neural" in examples
  assert "tier3_extras" in examples


def test_flax_nnx_get_doc_url() -> None:
  """Verify flax nnx get doc url behavior."""
  adapter: FlaxNNXAdapter = FlaxNNXAdapter()
  url = adapter.get_doc_url("flax.nnx.Linear")
  assert "flax.nnx.Linear" in url  # type: ignore[operator]
