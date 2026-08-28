"""Tests for the Jax framework adapter extra features."""

import pytest
from unittest.mock import patch
from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.jax import JaxCoreAdapter
from ml_switcheroo.enums import SemanticTier
from typing import Dict, List, Optional
import types


def test_jax_extra(monkeypatch: pytest.MonkeyPatch) -> None:
  """Test various JAX adapter functionalities like traits and semantic collection."""
  import sys

  # 66: Test jax not installed, no snapshot
  monkeypatch.setitem(sys.modules, "jax", None)

  # Mock load_snapshot_for_adapter
  import ml_switcheroo.frameworks.jax as jax_mod

  monkeypatch.setattr(jax_mod, "load_snapshot_for_adapter", lambda fw: {})

  adapter: JaxCoreAdapter = JaxCoreAdapter()

  # 165: plugin_traits
  traits = adapter.plugin_traits
  assert traits.has_numpy_compatible_arrays is True

  # 203-206: _collect_ghost
  # first test with empty snapshot
  res_ghost: List[Dict[str, str]] = adapter._collect_ghost(SemanticTier.LOSS)
  assert res_ghost == []

  # now with mock snapshot
  adapter._snapshot_data = {
    "categories": {SemanticTier.LOSS.value: [{"name": "foo", "api_path": "foo", "kind": "function"}]}
  }
  res_ghost = adapter._collect_ghost(SemanticTier.LOSS)
  assert len(res_ghost) == 1

  # 217-224: _collect_live
  res_live = adapter._collect_live(SemanticTier.LOSS)
  assert isinstance(res_live, list)

  res_live_opt = adapter._collect_live(SemanticTier.OPTIMIZER)
  assert isinstance(res_live_opt, list)

  res_live_act = adapter._collect_live(SemanticTier.ACTIVATION)
  assert isinstance(res_live_act, list)


def test_jax_adapter_ghost_init_empty() -> None:
  """Test element."""
  with patch("ml_switcheroo.frameworks.jax.jax", None):
    with patch("ml_switcheroo.frameworks.jax.load_snapshot_for_adapter", return_value={}):
      adapter: JaxCoreAdapter = JaxCoreAdapter()
      assert adapter._mode == InitMode.GHOST


def test_jax_adapter_properties() -> None:
  """Test element."""
  adapter: JaxCoreAdapter = JaxCoreAdapter()
  assert adapter.import_alias == ("jax.numpy", "jnp")
  assert "jax.numpy" in adapter.import_namespaces
  assert "import" in adapter.test_config
  assert "import jax" in adapter.harness_imports
  assert "jax.random.PRNGKey" in adapter.get_harness_init_code()
  pass
  adapter.get_device_syntax("cpu")
  adapter.get_device_syntax("gpu", "1")
  adapter.get_device_check_syntax()
  adapter.get_serialization_imports()
  adapter.get_serialization_syntax("load", "path")
  adapter.get_serialization_syntax("save", "path")
  adapter.get_serialization_syntax("other", "path")
  adapter.get_weight_conversion_imports()
  adapter.get_weight_load_code("path")
  adapter.get_tensor_to_numpy_expr("t")
  adapter.get_weight_save_code("t", "path")
  assert "jax.numpy.add.html" in adapter.get_doc_url("jax.numpy.add") or ""
  assert "optax" in adapter.get_tiered_examples()["tier3_extras"]


def test_jax_missing_methods() -> None:
  """Test element."""
  adapter: JaxCoreAdapter = JaxCoreAdapter()
  adapter.declared_magic_args
  adapter.rng_seed_methods
  adapter.structural_traits
  adapter.plugin_traits
  adapter.definitions
  adapter.convert([1, 2, 3])
  adapter.apply_wiring({})


def test_jax_convert_exception() -> None:
  """Test element."""
  adapter: JaxCoreAdapter = JaxCoreAdapter()
  real_import = __builtins__["__import__"]

  def mock_import(
    name: str,
    globals: Optional[Dict[str, str]] = None,
    locals: Optional[Dict[str, str]] = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
  ) -> types.ModuleType:
    if name == "jax.numpy":
      raise ImportError("mock")
    return real_import(name, globals, locals, fromlist, level)

  with patch("builtins.__import__", side_effect=mock_import):
    assert adapter.convert([1, 2, 3]) == [1, 2, 3]
