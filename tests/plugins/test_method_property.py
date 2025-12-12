import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.method_property import transform_method_to_property


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["method_to_property"] = transform_method_to_property
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()

  # Define mappings for test cases
  size_def = {"variants": {"jax": {"api": "shape", "requires_plugin": "method_to_property"}}}
  data_ptr_def = {"variants": {"jax": {"api": "data", "requires_plugin": "method_to_property"}}}

  all_defs = {"size": size_def, "data_ptr": data_ptr_def}

  # Update: rewriter.py calls _get_qualified_name which might resolve to just "size"
  # if 'x' is not a known module alias. To allow the plugin to trigger on generic calls,
  # we must ensure get_definition returns the mapping for "size".
  # In a real app, this happens if "size" is in the Spec or mapped via heuristics.
  def get_def_side_effect(name):
    if name == "size" or name.endswith(".size"):
      return ("size", size_def)
    # Rewriter core might not call get_definition for plugin internals,
    # but let's be consistent.
    return None

  mgr.get_definition.side_effect = get_def_side_effect

  # HookContext.lookup_api uses get_known_apis
  mgr.get_known_apis.return_value = all_defs

  # Helper for resolving variants
  def resolve_variant_side_effect(aid, fw):
    if aid in all_defs:
      return all_defs[aid]["variants"].get(fw)
    return None

  mgr.resolve_variant.side_effect = resolve_variant_side_effect
  mgr.is_verified.return_value = True

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_simple_size_conversion(rewriter):
  # This tests the rewriter flow calling the plugin
  # Note: Input must match what rewriter expects if it does a lookup first.
  # Here we test the plugin logic output passed through rewriter.
  # The plugin extracts method name "size".
  assert "x.shape" in rewrite_code(rewriter, "s = x.size()")


def test_indexed_size_conversion(rewriter):
  assert "x.shape[0]" in rewrite_code(rewriter, "d = x.size(0)").replace(" ", "")


def test_ignore_other_methods(rewriter):
  # 'other' is not in all_defs, so plugin returns original node
  assert "x.other()" in rewrite_code(rewriter, "x.other()")


def test_data_ptr_mapping(rewriter):
  # Direct hook invoke test verifying data-driven logic
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("data_ptr")))

  # The hook uses ctx to look up "data_ptr" in semantics (mocked in fixture)
  res = transform_method_to_property(node, rewriter.ctx)

  # Should convert to .data based on JSON mapping, NOT fallback code
  assert isinstance(res, cst.Attribute)
  assert res.attr.value == "data"
