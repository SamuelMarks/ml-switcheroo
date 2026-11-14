"""Test suite for the Jax Decompose module."""

import typing
from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.jax_decompose import decompose_via_jax


def test_jax_decompose_plugin() -> None:
  """Verify `Call` handles JAX decompose plugin as expected."""
  node: cst.Call = cst.Call(func=cst.Name("UnsupportedOp"), args=[])
  mock_config: RuntimeConfig = RuntimeConfig(target_framework="keras", source_framework="torch")
  mock_semantics: MagicMock = MagicMock()
  ctx: HookContext = HookContext(semantics=mock_semantics, config=mock_config)
  ctx.current_op_id = "Hardswish"
  result: cst.CSTNode | cst.Call = decompose_via_jax(node, ctx)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "hardswish"
  assert isinstance(result.func.value, cst.Attribute)
  assert result.func.value.attr.value == "numpy"


def test_jax_decompose_plugin_no_op_id() -> None:
  """Verify `Call` handles JAX decompose plugin no operation id as expected."""
  node: cst.Call = cst.Call(func=cst.Name("UnsupportedOp"), args=[])
  mock_config: RuntimeConfig = RuntimeConfig(target_framework="keras", source_framework="torch")
  mock_semantics: MagicMock = MagicMock()
  ctx: HookContext = HookContext(semantics=mock_semantics, config=mock_config)
  ctx.current_op_id = None
  result: cst.CSTNode | cst.Call = decompose_via_jax(node, ctx)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Attribute)
  assert result.func.attr.value == "unknownop"


def test_jax_decompose_discovery_branches(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify configured namespace, adapter import_alias fallback, and empty registry abort.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  from ml_switcheroo.enums import SemanticTier
  from ml_switcheroo.frameworks.base import ImportConfig
  from ml_switcheroo.plugins import jax_decompose as jd
  from ml_switcheroo.semantics.schema import PluginTraits

  node = cst.Call(func=cst.Name("CustomOp"), args=[])
  cfg_ns = RuntimeConfig(
    source_framework="torch",
    target_framework="keras",
    plugin_settings={"decompose_namespace": "custom.array_ir"},
  )
  ctx_ns = HookContext(semantics=None, config=cfg_ns)
  ctx_ns.current_op_id = "Sin"
  res_ns = decompose_via_jax(node, ctx_ns)
  assert cst.Module(body=[]).code_for_node(res_ns) == "custom.array_ir.sin()"

  class _AliasOnlyAdapter:
    """Adapter exposing ARRAY_API via import_alias fallback."""

    plugin_traits = PluginTraits(has_numpy_compatible_arrays=True, requires_explicit_rng=True)
    inherits_from = None
    import_namespaces: typing.ClassVar[dict[str, ImportConfig]] = {"other.ns": ImportConfig(tier=SemanticTier.NEURAL)}
    import_alias = ("alias_ir.np", "anp")

  class _BrokenAdapter:
    """Adapter missing plugin_traits to test AttributeError skip."""

  class _InvalidAliasAdapter:
    """Adapter with non-identifier alias string to test validation skip."""

    plugin_traits = PluginTraits(has_numpy_compatible_arrays=True, requires_explicit_rng=True)
    inherits_from = None
    import_namespaces: typing.ClassVar[dict[str, ImportConfig]] = {}
    import_alias = ("not-valid!", "bad")

  adapters: dict[str, _AliasOnlyAdapter | _BrokenAdapter | _InvalidAliasAdapter | None] = {
    "a_none": None,
    "b_broken": _BrokenAdapter(),
    "c_invalid": _InvalidAliasAdapter(),
    "d_alias": _AliasOnlyAdapter(),
  }
  monkeypatch.setattr(jd, "available_frameworks", lambda: list(adapters.keys()))
  monkeypatch.setattr(jd, "get_adapter", lambda name: adapters.get(name))

  ctx_alias = HookContext(
    semantics=None,
    config=RuntimeConfig(source_framework="torch", target_framework="keras"),
  )
  ctx_alias.current_op_id = "Cos"
  res_alias = decompose_via_jax(node, ctx_alias)
  assert cst.Module(body=[]).code_for_node(res_alias) == "alias_ir.np.cos()"

  monkeypatch.setattr(jd, "available_frameworks", lambda: ["a_none", "b_broken", "c_invalid"])
  assert decompose_via_jax(node, ctx_alias) is node
