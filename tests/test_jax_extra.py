"""Tests for the Jax framework adapter extra features."""

from ml_switcheroo.frameworks.jax import JaxCoreAdapter
from ml_switcheroo.enums import SemanticTier


def test_jax_extra(monkeypatch):
  """Test various JAX adapter functionalities like traits and semantic collection."""
  import sys

  # 66: Test jax not installed, no snapshot
  monkeypatch.setitem(sys.modules, "jax", None)

  # Mock load_snapshot_for_adapter
  import ml_switcheroo.frameworks.jax as jax_mod

  monkeypatch.setattr(jax_mod, "load_snapshot_for_adapter", lambda fw: {})

  adapter = JaxCoreAdapter()

  # 165: plugin_traits
  traits = adapter.plugin_traits
  assert traits.has_numpy_compatible_arrays is True

  # 203-206: _collect_ghost
  # first test with empty snapshot
  res_ghost = adapter._collect_ghost(SemanticTier.LOSS)
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
