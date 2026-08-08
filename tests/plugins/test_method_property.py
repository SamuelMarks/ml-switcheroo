"""Test suite for the Method Property module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.method_property import transform_method_to_property


def rewrite_code(rewriter, code):
  """Rewrites code."""
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  hooks._HOOKS["method_to_property"] = transform_method_to_property
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  size_def = {"variants": {"jax": {"api": "shape", "requires_plugin": "method_to_property"}}}
  data_ptr_def = {"variants": {"jax": {"api": "data", "requires_plugin": "method_to_property"}}}
  all_defs = {"size": size_def, "data_ptr": data_ptr_def}

  def get_def_side_effect(name):
    """Gets def side effect."""
    if name == "size" or name.endswith(".size"):
      return ("size", size_def)
    return None

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.get_known_apis.return_value = all_defs

  def resolve_variant_side_effect(aid, fw):
    """Resolves variant side effect."""
    if aid in all_defs:
      return all_defs[aid]["variants"].get(fw)
    return None

  mgr.resolve_variant.side_effect = resolve_variant_side_effect
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_simple_size_conversion(rewriter):
  """Verifies the behavior of simple size conversion."""
  assert "x.shape" in rewrite_code(rewriter, "s = x.size()")


def test_indexed_size_conversion(rewriter):
  """Verifies the behavior of indexed size conversion."""
  assert "x.shape[0]" in rewrite_code(rewriter, "d = x.size(0)").replace(" ", "")


def test_ignore_other_methods(rewriter):
  """Verifies the behavior of ignore other methods."""
  assert "x.other()" in rewrite_code(rewriter, "x.other()")


def test_obj_type_not_tensor(rewriter):
  """Verifies that the method is not rewritten if the receiver is known to not be a tensor."""
  rewriter.ctx.resolve_type = MagicMock(return_value="Module")
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("size")))
  res = transform_method_to_property(node, rewriter.ctx)
  assert res is node


def test_missing_target_prop(rewriter):
  """Verifies that the method is not rewritten if lookup_api fails."""
  rewriter.ctx.resolve_type = MagicMock(return_value="Tensor")
  rewriter.ctx.lookup_api = MagicMock(return_value=None)
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("size")))
  res = transform_method_to_property(node, rewriter.ctx)
  assert res is node


def test_too_many_args(rewriter):
  """Verifies that the method is not rewritten if it has multiple args."""
  rewriter.ctx.resolve_type = MagicMock(return_value="Tensor")
  rewriter.ctx.lookup_api = MagicMock(return_value="shape")
  node = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("size")),
    args=[cst.Arg(cst.Integer("0")), cst.Arg(cst.Integer("1"))],
  )
  res = transform_method_to_property(node, rewriter.ctx)
  assert res is node


def test_data_ptr_mapping(rewriter):
  """Verifies the behavior of data ptr mapping."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("data_ptr")))
  res = transform_method_to_property(node, rewriter.ctx)
  assert isinstance(res, cst.Attribute)
  assert res.attr.value == "data"


def test_func_not_attribute(rewriter):
  """Verifies the behavior when node.func is not an Attribute."""
  node = cst.Call(func=cst.Name("size"))
  res = transform_method_to_property(node, rewriter.ctx)
  assert res is node


def test_unknown_method(rewriter):
  """Verifies the behavior when the method name is not recognized."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("unknown_method")))
  res = transform_method_to_property(node, rewriter.ctx)
  assert res is node
