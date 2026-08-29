"""Test suite for the Inplace Unroll module."""

import typing
from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.inplace_unroll import _get_method_name, _get_receiver_name, unroll_inplace_ops
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code."""
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return typing.cast(str, new_tree.code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Docstring."""
  hooks._HOOKS["unroll_inplace_ops"] = unroll_inplace_ops
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()

  def get_def(name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Gets def."""
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
  """Verifies the behavior of strip inplace underscore."""
  code: str = "res = x.add_(y)"
  result: str = rewrite_code(rewriter, code)
  assert result == "res = x + y"


def test_metadata_trigger_implicit(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of metadata trigger implicit."""
  mock_hook = MagicMock(return_value=cst.Name("HookRan"))
  hooks._HOOKS["unroll_inplace_ops"] = mock_hook
  code: str = "x.assign_add(y)"
  rewrite_code(rewriter, code)
  mock_hook.assert_called_once()


def test_fallback_non_math_unroll(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of fallback non math unroll."""
  code: str = "res = x.custom_(y)"
  result: str = rewrite_code(rewriter, code)
  assert "x.custom(y)" in result


def test_ignore_standard_calls(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of ignore standard calls."""
  code: str = "x.add(y)"
  res: str = rewrite_code(rewriter, code)
  assert res == code


def test_ignore_dunders(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of ignore dunders."""
  hook = unroll_inplace_ops
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("__init__")))
  res: typing.Any = hook(node, None)
  assert typing.cast(cst.Name, res.func.attr).value == "__init__"


def test_ignore_single_underscore(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of ignore single underscore."""
  hook = unroll_inplace_ops
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("_")))
  res: typing.Any = hook(node, None)
  assert typing.cast(cst.Name, res.func.attr).value == "_"


# --- Merged from test_inplace_unroll_extra.py ---


def test_inplace_unroll_not_attribute() -> None:
  """Docstring."""
  node = cst.parse_expression("add_(x)")
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics_mock = MagicMock()
  ctx = HookContext(semantics=semantics_mock, config=config)

  res: typing.Any = unroll_inplace_ops(typing.cast(cst.Call, node), ctx)
  assert res is node


def test_get_receiver_name_not_attribute() -> None:
  """Docstring."""
  node = cst.parse_expression("add_(x)")
  assert _get_receiver_name(typing.cast(cst.Call, node)) is None


def test_get_method_name_not_attribute() -> None:
  """Docstring."""
  node = cst.parse_expression("add_(x)")
  assert _get_method_name(typing.cast(cst.Call, node)) is None


def test_inplace_unroll_just_underscore() -> None:
  """Docstring."""
  node = cst.parse_expression("x._(y)")
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics_mock = MagicMock()
  ctx = HookContext(semantics=semantics_mock, config=config)

  res: typing.Any = unroll_inplace_ops(typing.cast(cst.Call, node), ctx)
  assert res is node


def test_inplace_unroll_no_underscore() -> None:
  """Docstring."""
  node = cst.parse_expression("x.add(y)")
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics_mock = MagicMock()
  ctx = HookContext(semantics=semantics_mock, config=config)

  res: typing.Any = unroll_inplace_ops(typing.cast(cst.Call, node), ctx)
  assert res is node
