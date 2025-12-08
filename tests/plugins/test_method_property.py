"""
Tests for the Method-to-Property Conversion Plugin.

Verifies that:
1. `x.size()` converts to `x.shape`.
2. `x.size(0)` converts to `x.shape[0]`.
3. Fallback logic handles unknowns gracefully.
"""

import pytest
from unittest.mock import MagicMock
import libcst as cst

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.method_property import transform_method_to_property


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  tree = cst.parse_module(code)
  new_tree = tree.visit(rewriter)
  return new_tree.code


@pytest.fixture
def rewriter():
  # 1. Register Hook
  hooks._HOOKS["method_to_property"] = transform_method_to_property
  hooks._PLUGINS_LOADED = True

  # 2. Mock Semantics
  mgr = MagicMock()

  # Define mappings for test cases
  size_def = {
    "requires_plugin": "method_to_property",
    "std_args": ["x"],
    "variants": {
      "torch": {"api": "torch.Tensor.size"},
      "jax": {"api": "shape", "requires_plugin": "method_to_property"},
    },
  }

  def get_def_side_effect(name):
    # Match generic or specific usage patterns in code
    if "size" in name:
      return "size", size_def
    return None

  mgr.get_definition.side_effect = get_def_side_effect
  # Mock API lookup used in hook: ctx.lookup_api("size")
  # This lookup goes through semantics manager for the target fw variant
  # Simulate finding "shape" for "size"
  mgr.get_known_apis.return_value = {"size": size_def}

  mgr.is_verified.return_value = True

  # 3. Config
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(semantics=mgr, config=cfg)


def test_simple_size_conversion(rewriter):
  """
  Scenario: `x.size()`
  Expectation: `x.shape`
  """
  code = "s = x.size()"
  result = rewrite_code(rewriter, code)

  assert "x.shape" in result
  assert "size()" not in result


def test_indexed_size_conversion(rewriter):
  """
  Scenario: `x.size(0)`
  Expectation: `x.shape[0]`
  """
  code = "dim0 = x.size(0)"
  result = rewrite_code(rewriter, code)

  # Check subscript structure in string output
  # Note: LibCST output formatting might vary slightly with whitespace
  clean = result.replace(" ", "")
  assert "x.shape[0]" in clean


def test_ignore_other_methods(rewriter):
  """
  Scenario: `x.other()`
  Expectation: No change.
  """
  # Ensure the mock lookup returns None for "other"
  code = "val = x.other()"
  result = rewrite_code(rewriter, code)

  assert "x.other()" in result


def test_data_ptr_fallback_logic(rewriter):
  """
  Scenario: `x.data_ptr()` -> `x.data` (if configured via fallback logic in plugin).
  """
  # We test the hardcoded fallback for "data_ptr" in the plugin logic
  # even if semantics lookup fails (simulating missing JSON entry).

  # Note: In the mock, lookup_api returns keys from get_known_apis().
  # We didn't add data_ptr there, so lookup returns None.
  # The plugin should fallback.

  # However, rewriter only calls hook if semantics says requires_plugin.
  # So we MUST have a semantic entry to trigger the hook call in the first place.

  # Manually invoke hook to test internal logic without rewriter dispatch
  call_node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("data_ptr")))

  new_node = transform_method_to_property(call_node, rewriter.ctx)

  # Should transform to x.data
  assert isinstance(new_node, cst.Attribute)
  assert new_node.attr.value == "data"
