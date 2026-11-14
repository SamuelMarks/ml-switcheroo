"""Unit tests for `JaxCoreAdapter` (`ml_switcheroo.frameworks.jax`)."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest
from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import JsonValue

import ml_switcheroo.frameworks.jax as jax_mod
from ml_switcheroo.core.ghost import GhostRef
from ml_switcheroo.frameworks.base import FrameworkSnapshot, ImportConfig, InitMode
from ml_switcheroo.frameworks.jax import JaxCoreAdapter


def test_jax_adapter_metadata_traits_and_syntax() -> None:
  """Verify `JaxCoreAdapter` metadata, namespaces, traits, device/serialization/weight syntax, and docs."""
  adapter = JaxCoreAdapter()
  assert adapter.display_name == "JAX (no framework)"
  assert adapter.inherits_from is None
  assert adapter.ui_priority == 10
  assert adapter.import_alias == ("jax.numpy", "jnp")

  ns = adapter.import_namespaces
  jnp_ns = ns["jax.numpy"]
  assert isinstance(jnp_ns, ImportConfig)
  assert jnp_ns.recommended_alias == "jnp"
  assert "optax" in ns
  assert "import jax.numpy as jnp" in adapter.test_config["import"]
  assert "import jax" in adapter.harness_imports
  assert "import jax.random" in adapter.harness_imports

  init_code = adapter.get_harness_init_code()
  assert "def _make_jax_key(seed):" in init_code
  assert "jax.random.PRNGKey(seed)" in init_code
  assert adapter.declared_magic_args == ["key"]
  assert adapter.rng_seed_methods == []

  s_traits = adapter.structural_traits
  assert s_traits.module_base is None
  assert s_traits.forward_method == "__call__"
  assert s_traits.requires_super_init is False
  assert adapter.plugin_traits.has_numpy_compatible_arrays is True
  assert isinstance(adapter.definitions, dict)

  assert "jax.devices" in adapter.get_device_syntax("cpu")
  assert "jax.devices" in adapter.get_device_syntax("gpu", "1")
  assert len(adapter.get_device_check_syntax()) > 0
  assert len(adapter.get_serialization_imports()) > 0
  assert len(adapter.get_serialization_syntax("load", "path")) > 0
  assert len(adapter.get_serialization_syntax("save", "path", "state")) > 0
  assert adapter.get_serialization_syntax("other", "path") == ""
  assert len(adapter.get_weight_conversion_imports()) > 0
  assert len(adapter.get_weight_load_code("path")) > 0
  assert len(adapter.get_tensor_to_numpy_expr("t")) > 0
  assert len(adapter.get_weight_save_code("t", "path")) > 0

  assert adapter.get_doc_url("jax.numpy.abs") == "https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.abs.html"
  examples = adapter.get_tiered_examples()
  assert {"tier1_math", "tier2_neural", "tier3_extras", "tier4_qwen3-vl"} <= set(examples.keys())
  assert "optax" in examples["tier3_extras"]

  snapshot: dict[str, dict[str, JsonValue]] = {}
  adapter.apply_wiring(snapshot)
  assert "mappings" in snapshot
  assert "templates" in snapshot


def test_jax_adapter_modes_collection_and_convert(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify LIVE/GHOST initialization, `_collect_ghost`/`_collect_live`, module reload, and `convert` fallbacks.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  # LIVE mode
  monkeypatch.setattr(jax_mod, "jax", True)
  live_adapter = JaxCoreAdapter()
  assert live_adapter._mode == InitMode.LIVE

  # GHOST mode without and with snapshot categories
  with (
    patch.object(jax_mod, "jax", None),
    patch.object(jax_mod, "load_snapshot_for_adapter", return_value={}),
  ):
    ghost_empty = JaxCoreAdapter()
    assert ghost_empty._mode == InitMode.GHOST
    assert ghost_empty._collect_ghost(SemanticTier.LOSS) == []

  with (
    patch.object(jax_mod, "jax", None),
    patch.object(
      jax_mod,
      "load_snapshot_for_adapter",
      return_value={"categories": {"loss": [{"name": "foo", "kind": "function", "api_path": "foo"}]}},
    ),
  ):
    ghost_snap = JaxCoreAdapter()
    assert ghost_snap._mode == InitMode.GHOST
    ghosts = ghost_snap._collect_ghost(SemanticTier.LOSS)
    assert len(ghosts) == 1
    assert ghosts[0].name == "foo"

  ghost_snap._snapshot_data = FrameworkSnapshot()
  assert ghost_snap._collect_ghost(SemanticTier.LOSS) == []

  # `_collect_live` across LOSS, OPTIMIZER, ACTIVATION, and unsupported tier
  loss_ref = GhostRef(name="l2", api_path="optax.l2_loss", kind="function")
  opt_ref = GhostRef(name="adam", api_path="optax.adam", kind="function")
  assert isinstance(live_adapter._collect_live(SemanticTier.LOSS), list)
  assert isinstance(live_adapter._collect_live(SemanticTier.OPTIMIZER), list)
  assert isinstance(live_adapter._collect_live(SemanticTier.ACTIVATION), list)
  assert live_adapter._collect_live(SemanticTier.ARRAY_API) == []

  with patch.object(JaxCoreAdapter, "_scan_optax_losses", return_value=[loss_ref], create=True):
    assert live_adapter._collect_live(SemanticTier.LOSS) == [loss_ref]
  with patch.object(JaxCoreAdapter, "_scan_optax_optimizers", return_value=[opt_ref], create=True):
    assert live_adapter._collect_live(SemanticTier.OPTIMIZER) == [opt_ref]

  # `convert`: non-array passthrough, `jnp.array` conversion, ValueError fallback, and ImportError fallback
  assert live_adapter.convert("test") == "test"
  assert live_adapter.convert(42) == 42

  mock_jax = MagicMock()
  mock_jnp = MagicMock()
  mock_jax.numpy = mock_jnp
  mock_jnp.array.return_value = "jax_arr"
  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "jax", mock_jax)
    mp.setitem(sys.modules, "jax.numpy", mock_jnp)
    assert live_adapter.convert([1, 2, 3]) == "jax_arr"
    mock_jnp.array.side_effect = ValueError("bad array")
    assert live_adapter.convert([1, 2, 3]) == [1, 2, 3]

  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "jax", None)
    mp.setitem(sys.modules, "jax.numpy", None)
    assert live_adapter.convert([1, 2, 3]) == [1, 2, 3]
    importlib.reload(jax_mod)
    assert jax_mod.jax is None
    assert jax_mod.jnp is None

  with pytest.MonkeyPatch.context() as mp:
    mp.setitem(sys.modules, "jax", mock_jax)
    mp.setitem(sys.modules, "jax.numpy", mock_jnp)
    importlib.reload(jax_mod)
    assert JaxCoreAdapter()._mode == InitMode.LIVE

  importlib.reload(jax_mod)
