"""
Tests for In-Place Operation Unrolling Plugin.

Verifies:
1. Detection of `_` suffixed methods.
2. Stripping of the underscore to map to functional equivalents.
3. Ignoring of dunder methods or non-inplace calls.
4. **NEW**: Detection via `is_inplace` metadata even without underscore.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

# Fix: Import TestRewriter shim
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.inplace_unroll import unroll_inplace_ops
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Executes pipeline."""
  tree = cst.parse_module(code)
  # Use .convert()
  new_tree = rewriter.convert(tree)
  return new_tree.code


@pytest.fixture
def rewriter():
  # 1. Register Hook
  hooks._HOOKS["unroll_inplace_ops"] = unroll_inplace_ops
  hooks._PLUGINS_LOADED = True

  # 2. Mock Semantics
  mgr = MagicMock()

  def get_def(name):
    if "add_" in name or "sub_" in name:
      base = name.split(".")[-1]
      return (
        base,
        {
          "variants": {
            "torch": {"api": name},
            "jax": {"requires_plugin": "unroll_inplace_ops"},
          }
        },
      )

    # Feature Test: Op marked inplace but no underscore (e.g. 'assign_add')
    if "assign_add" in name:
      return ("AssignAdd", {"is_inplace": True, "variants": {"target": {}}})

    return None

  # Important: Hook Context logic often uses get_definition to populate metadata
  # handle_pre_checks specifically calls it

  mgr.get_definition.side_effect = get_def
  mgr.is_verified.return_value = True
  # Mock get_mapping called in pre.py
  mgr.resolve_variant.return_value = {}

  # 3. Config
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  # Fix: Use positional arguments for initialization
  return PivotRewriter(mgr, cfg)


def test_strip_inplace_underscore(rewriter):
  """
  Scenario: `x.add_(y)`
  Expectation: `x + y` (Infix operator for JAX compatibility).
  """
  code = "res = x.add_(y)"
  # We test the full rewriter which calls pre-checks -> hook -> logic
  result = rewrite_code(rewriter, code)

  # Plugin transforms add_ to +
  assert result == "res = x + y"


def test_metadata_trigger_implicit(rewriter):
  """
  Scenario: Op 'assign_add' has no underscore, but ODL metadata says `is_inplace=True`.
  Expectation: Plugin runs, stripping suffix (idempotent if no suffix? no, plugin expects _)

  Note: The actual plugin requires underscore by implementation details.
  This test verifies that the rewriter *attempts* to call the hook if metadata is set.
  """

  # To verifying wiring, we mock the hook itself in the test to see if called.
  mock_hook = MagicMock(return_value=cst.Name("HookRan"))

  # Monkeypatch the hook reference used by pre.py
  # NOTE: pre.py uses `get_hook`. We need to register into _HOOKS.
  hooks._HOOKS["unroll_inplace_ops"] = mock_hook

  code = "x.assign_add(y)"
  # Trigger traversal
  rewrite_code(rewriter, code)

  mock_hook.assert_called_once()


def test_fallback_non_math_unroll(rewriter):
  """
  Scenario: `x.custom_(y)` (Unknown op, matched via heuristics in pre.py)
  Expectation: `x.custom(y)` (Method strip fallback).
  """
  # This relies on the 'endswith(_)' check in pre.py
  code = "res = x.custom_(y)"
  result = rewrite_code(rewriter, code)

  assert "x.custom(y)" in result


def test_ignore_standard_calls(rewriter):
  """
  Scenario: `x.add(y)` (No underscore, no metadata)
  Expectation: No change.
  """
  code = "x.add(y)"
  res = rewrite_code(rewriter, code)
  assert res == code


def test_ignore_dunders(rewriter):
  """
  Scenario: `x.__init__(y)`
  Expectation: No stripping of underscore.
  """
  hook = unroll_inplace_ops
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("__init__")))
  # Direct hook call
  res = hook(node, None)
  # Identify by attribute name
  assert res.func.attr.value == "__init__"


def test_ignore_single_underscore(rewriter):
  """
  Scenario: `x._(y)` (Obscure but valid python)
  Expectation: No strip.
  """
  hook = unroll_inplace_ops
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("_")))
  res = hook(node, None)
  assert res.func.attr.value == "_"
