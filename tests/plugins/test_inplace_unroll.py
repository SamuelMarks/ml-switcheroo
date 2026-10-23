"""Test suite for the Inplace Unroll module."""

from pydantic import JsonValue
from unittest.mock import MagicMock

import libcst as cst
import pytest

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
  """Execute rewriter helper logic.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  hooks._HOOKS["unroll_inplace_ops"] = unroll_inplace_ops
  hooks._PLUGINS_LOADED = True
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
  """Verifies the behavior of strip inplace underscore.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "res = x.add_(y)"
  result: str = rewrite_code(rewriter, code)
  assert result == "res = x + y"


def test_metadata_trigger_implicit(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of metadata trigger implicit.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  mock_hook = MagicMock(return_value=cst.Name("HookRan"))
  hooks._HOOKS["unroll_inplace_ops"] = mock_hook
  code: str = "x.assign_add(y)"
  rewrite_code(rewriter, code)
  mock_hook.assert_called_once()


def test_fallback_non_math_unroll(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of fallback non math unroll.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "res = x.custom_(y)"
  result: str = rewrite_code(rewriter, code)
  assert "x.custom(y)" in result


def test_ignore_standard_calls(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of ignore standard calls.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  code: str = "x.add(y)"
  res: str = rewrite_code(rewriter, code)
  assert res == code


def test_ignore_dunders(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of ignore dunders.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  hook = unroll_inplace_ops
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("__init__")))
  res = getattr(
    {
      "v": getattr({"v": hook}, "__getitem__")("v")(
        getattr({"v": node}, "__getitem__")("v"), getattr({"v": None}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert getattr(
    {
      "v": getattr(
        {
          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
            getattr(
              {"v": getattr({"v": getattr({"v": res}, "__getitem__")("v").func}, "__getitem__")("v").attr}, "__getitem__"
            )("v"),
            getattr({"v": getattr({"v": cst}, "__getitem__")("v").Name}, "__getitem__")("v"),
          )
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "__init__"}, "__getitem__")("v")


def test_ignore_single_underscore(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of ignore single underscore.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  hook = unroll_inplace_ops
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("_")))
  res = getattr(
    {
      "v": getattr({"v": hook}, "__getitem__")("v")(
        getattr({"v": node}, "__getitem__")("v"), getattr({"v": None}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  assert getattr(
    {
      "v": getattr(
        {
          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
            getattr(
              {"v": getattr({"v": getattr({"v": res}, "__getitem__")("v").func}, "__getitem__")("v").attr}, "__getitem__"
            )("v"),
            getattr({"v": getattr({"v": cst}, "__getitem__")("v").Name}, "__getitem__")("v"),
          )
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "_"}, "__getitem__")("v")


def test_inplace_unroll_not_attribute() -> None:
  """Verify inplace unroll not attribute behavior."""
  node = cst.parse_expression("add_(x)")
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics_mock = MagicMock()
  ctx = HookContext(semantics=semantics_mock, config=config)

  res = unroll_inplace_ops(cst.ensure_type(node, cst.Call), ctx)
  assert res is node


def test_get_receiver_name_not_attribute() -> None:
  """Verify get receiver name not attribute behavior."""
  node = cst.parse_expression("add_(x)")
  assert _get_receiver_name(cst.ensure_type(node, cst.Call)) is None


def test_get_method_name_not_attribute() -> None:
  """Verify get method name not attribute behavior."""
  node = cst.parse_expression("add_(x)")
  assert _get_method_name(cst.ensure_type(node, cst.Call)) is None


def test_inplace_unroll_just_underscore() -> None:
  """Verify inplace unroll just underscore behavior."""
  node = cst.parse_expression("x._(y)")
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics_mock = MagicMock()
  ctx = HookContext(semantics=semantics_mock, config=config)

  res = unroll_inplace_ops(cst.ensure_type(node, cst.Call), ctx)
  assert res is node


def test_inplace_unroll_no_underscore() -> None:
  """Verify inplace unroll no underscore behavior."""
  node = cst.parse_expression("x.add(y)")
  config = RuntimeConfig(source_framework="torch", target_framework="jax")
  semantics_mock = MagicMock()
  ctx = HookContext(semantics=semantics_mock, config=config)

  res = unroll_inplace_ops(cst.ensure_type(node, cst.Call), ctx)
  assert res is node
