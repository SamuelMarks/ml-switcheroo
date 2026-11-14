"""Test suite for the Inplace Unroll module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest
from pydantic import JsonValue

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.inplace_unroll import _get_method_name, _get_receiver_name, unroll_inplace_ops
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      code: Python or DSL source code string to parse or transpile.
      rewriter: Configured CST rewriter test helper instance.

  Returns:
      Resulting `str` from rewrite code.
  """
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return str(new_tree.code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Create a configured `PivotRewriter` fixture for inplace unroll tests.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  hooks.register_hook("unroll_inplace_ops")(unroll_inplace_ops)
  hooks.set_plugins_loaded(True)
  mgr = MagicMock()

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get def.
    """
    if "add_" in name or "sub_" in name:
      base = name.split(".")[-1]
      return (base, {"variants": {"torch": {"api": name}, "jax": {"requires_plugin": "unroll_inplace_ops"}}})
    if "assign_add" in name:
      return ("AssignAdd", {"is_inplace": True, "variants": {"target": {}}})
    return None

  mgr.get_definition.side_effect = get_def
  mgr.is_verified.return_value = True
  mgr.resolve_variant.return_value = {}
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_strip_inplace_underscore(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` handles strip inplace underscore as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "res = x.add_(y)"
  result: str = rewrite_code(rewriter, code)
  assert result == "res = x + y"


def test_metadata_trigger_implicit(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` handles metadata trigger implicit as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  mock_hook = MagicMock(return_value=cst.Name("HookRan"))
  hooks.get_default_hook_registry().hooks["unroll_inplace_ops"] = mock_hook
  code: str = "x.assign_add(y)"
  rewrite_code(rewriter, code)
  mock_hook.assert_called_once()


def test_fallback_non_math_unroll(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `x.custom(y)` for fallback non math unroll.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "res = x.custom_(y)"
  result: str = rewrite_code(rewriter, code)
  assert "x.custom(y)" in result


def test_ignore_standard_calls(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` handles ignore standard calls as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "x.add(y)"
  res: str = rewrite_code(rewriter, code)
  assert res == code


def test_ignore_dunders(rewriter: PivotRewriter) -> None:
  """Verify `Call` handles ignore dunders as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  hook = unroll_inplace_ops
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("__init__")))
  res = hook(node, rewriter.ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Attribute)
  assert cst.ensure_type(res.func.attr, cst.Name).value == "__init__"


def test_ignore_single_underscore(rewriter: PivotRewriter) -> None:
  """Verify `Call` handles ignore single underscore as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  hook = unroll_inplace_ops
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("_")))
  res = hook(node, rewriter.ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Attribute)
  assert cst.ensure_type(res.func.attr, cst.Name).value == "_"


def test_inplace_unroll_not_attribute() -> None:
  """Verify `RuntimeConfig` handles inplace unroll not attribute as expected."""
  node = cst.parse_expression("add_(x)")
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics_mock = MagicMock()
  ctx = HookContext(semantics=semantics_mock, config=config)

  res = unroll_inplace_ops(cst.ensure_type(node, cst.Call), ctx)
  assert res is node


def test_get_receiver_name_not_attribute() -> None:
  """Verify `_get_receiver_name` handles not attribute as expected."""
  node = cst.parse_expression("add_(x)")
  assert _get_receiver_name(cst.ensure_type(node, cst.Call)) is None


def test_get_method_name_not_attribute() -> None:
  """Verify `_get_method_name` handles not attribute as expected."""
  node = cst.parse_expression("add_(x)")
  assert _get_method_name(cst.ensure_type(node, cst.Call)) is None


def test_inplace_unroll_just_underscore() -> None:
  """Verify `RuntimeConfig` handles inplace unroll just underscore as expected."""
  node = cst.parse_expression("x._(y)")
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics_mock = MagicMock()
  ctx = HookContext(semantics=semantics_mock, config=config)

  res = unroll_inplace_ops(cst.ensure_type(node, cst.Call), ctx)
  assert res is node


def test_inplace_unroll_no_underscore() -> None:
  """Verify `RuntimeConfig` handles inplace unroll no underscore as expected."""
  node = cst.parse_expression("x.add(y)")
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics_mock = MagicMock()
  ctx = HookContext(semantics=semantics_mock, config=config)

  res = unroll_inplace_ops(cst.ensure_type(node, cst.Call), ctx)
  assert res is node
