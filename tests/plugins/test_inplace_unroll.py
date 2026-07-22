"""Test suite for the Inplace Unroll module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.inplace_unroll import unroll_inplace_ops


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code."""
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return new_tree.code


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  hooks._HOOKS["unroll_inplace_ops"] = unroll_inplace_ops
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()

  def get_def(name):
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


def test_strip_inplace_underscore(rewriter):
  """Verifies the behavior of strip inplace underscore."""
  code = "res = x.add_(y)"
  result = rewrite_code(rewriter, code)
  assert result == "res = x + y"


def test_metadata_trigger_implicit(rewriter):
  """Verifies the behavior of metadata trigger implicit."""
  mock_hook = MagicMock(return_value=cst.Name("HookRan"))
  hooks._HOOKS["unroll_inplace_ops"] = mock_hook
  code = "x.assign_add(y)"
  rewrite_code(rewriter, code)
  mock_hook.assert_called_once()


def test_fallback_non_math_unroll(rewriter):
  """Verifies the behavior of fallback non math unroll."""
  code = "res = x.custom_(y)"
  result = rewrite_code(rewriter, code)
  assert "x.custom(y)" in result


def test_ignore_standard_calls(rewriter):
  """Verifies the behavior of ignore standard calls."""
  code = "x.add(y)"
  res = rewrite_code(rewriter, code)
  assert res == code


def test_ignore_dunders(rewriter):
  """Verifies the behavior of ignore dunders."""
  hook = unroll_inplace_ops
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("__init__")))
  res = hook(node, None)
  assert res.func.attr.value == "__init__"


def test_ignore_single_underscore(rewriter):
  """Verifies the behavior of ignore single underscore."""
  hook = unroll_inplace_ops
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("_")))
  res = hook(node, None)
  assert res.func.attr.value == "_"
