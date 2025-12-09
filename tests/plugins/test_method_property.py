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

  op_def = {"variants": {"jax": {"api": "shape", "requires_plugin": "method_to_property"}}}

  mgr.get_definition.return_value = ("size", op_def)
  mgr.get_known_apis.return_value = {"size": op_def}
  mgr.resolve_variant.side_effect = lambda aid, fw: op_def["variants"].get(fw)

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_simple_size_conversion(rewriter):
  assert "x.shape" in rewrite_code(rewriter, "s = x.size()")


def test_indexed_size_conversion(rewriter):
  assert "x.shape[0]" in rewrite_code(rewriter, "d = x.size(0)").replace(" ", "")


def test_ignore_other_methods(rewriter):
  # Mock returns None for 'other'
  rewriter.semantics.get_definition.side_effect = lambda x: ("size", {}) if "size" in x else None
  assert "x.other()" in rewrite_code(rewriter, "x.other()")


def test_data_ptr_fallback_logic(rewriter):
  # Direct hook invoke test
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("data_ptr")))
  res = transform_method_to_property(node, rewriter.ctx)
  assert res.attr.value == "data"
