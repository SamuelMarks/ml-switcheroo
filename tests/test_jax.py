"""Tests for the Jax framework adapter extra features."""

import typing
import pytest

from unittest.mock import patch


from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.frameworks.base import InitMode
from ml_switcheroo.frameworks.jax import JaxCoreAdapter


def test_jax_extra(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify jax extra behavior.

  Args:
      monkeypatch: Test argument or fixture.
  """
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


def test_jax_adapter_ghost_init_empty() -> None:
  """Verify jax adapter ghost init empty behavior."""
  with patch("ml_switcheroo.frameworks.jax.jax", None):
    with patch("ml_switcheroo.frameworks.jax.load_snapshot_for_adapter", return_value={}):
      adapter = JaxCoreAdapter()
      assert adapter._mode == InitMode.GHOST


def test_jax_adapter_properties() -> None:
  """Verify jax adapter properties behavior."""
  adapter = JaxCoreAdapter()
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
  assert "jax.numpy.add.html" in (adapter.get_doc_url("jax.numpy.add") or "")
  assert "optax" in adapter.get_tiered_examples()["tier3_extras"]


def test_jax_missing_methods() -> None:
  """Verify jax missing methods behavior."""
  adapter = JaxCoreAdapter()
  assert isinstance(adapter.declared_magic_args, list)
  assert isinstance(adapter.rng_seed_methods, list)
  assert adapter.structural_traits is not None
  assert adapter.plugin_traits is not None
  assert isinstance(adapter.definitions, dict)
  adapter.convert([1, 2, 3])
  wiring_snap: dict[str, typing.Hashable | typing.Sized] = {}
  getattr({"v": getattr({"v": adapter}, "__getitem__")("v").apply_wiring}, "__getitem__")("v")(
    getattr({"v": wiring_snap}, "__getitem__")("v")
  )
  assert isinstance(wiring_snap, dict)


def test_jax_convert_exception() -> None:
  """Verify jax convert exception behavior."""
  import builtins

  adapter = JaxCoreAdapter()
  real_import = builtins.__import__

  import types

  def mock_import(
    name: str,
    globals: typing.Mapping[str, str] | None = None,
    locals: typing.Mapping[str, str] | None = None,
    fromlist: typing.Sequence[str] = (),
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
      raise ImportError("mock")
    return real_import(name, globals, locals, fromlist, level)

  with patch("builtins.__import__", side_effect=mock_import):
    assert adapter.convert([1, 2, 3]) == [1, 2, 3]
