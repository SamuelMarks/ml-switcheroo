"""Test suite for the Casting module."""

import pytest
import libcst as cst
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.casting import transform_casting


def rewrite_call(rewriter, code):
  """Rewrites call."""
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  """Provides a mock rewriter for testing."""
  hooks._HOOKS["type_methods"] = transform_casting
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  cast_float_def = {
    "variants": {"torch": {"api": "torch.Tensor.float"}, "jax": {"api": "astype", "requires_plugin": "type_methods"}},
    "metadata": {"target_type": "Float32"},
  }
  cast_long_def = {
    "variants": {"torch": {"api": "torch.Tensor.long"}, "jax": {"api": "astype", "requires_plugin": "type_methods"}},
    "metadata": {"target_type": "Int64"},
  }
  float32_def = {"variants": {"jax": {"api": "jax.numpy.float32"}}}
  int64_def = {"variants": {"jax": {"api": "jax.numpy.int64"}}}
  all_defs = {"CastFloat": cast_float_def, "CastLong": cast_long_def, "Float32": float32_def, "Int64": int64_def}

  def get_def(name):
    """Gets def."""
    if "float" in name:
      return ("CastFloat", cast_float_def)
    if "long" in name:
      return ("CastLong", cast_long_def)
    return None

  def get_def_by_id(op_id):
    """Gets def by id."""
    return all_defs.get(op_id)

  def resolve(aid, fw):
    """Resolves ."""
    defn = all_defs.get(aid)
    if defn and fw in defn["variants"]:
      return defn["variants"][fw]
    return None

  mgr.get_definition.side_effect = get_def
  mgr.get_definition_by_id.side_effect = get_def_by_id
  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = all_defs
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": True}}
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_float_cast(rewriter):
  """Verifies the behavior of float cast."""
  rewriter.ctx.current_op_id = "CastFloat"
  code = "y = x.float()"
  res = rewrite_call(rewriter, code)
  assert ".astype" in res
  assert "jax.numpy.float32" in res


def test_long_cast(rewriter):
  """Verifies the behavior of long cast."""
  rewriter.ctx.current_op_id = "CastLong"
  code = "idx = mask.long()"
  res = rewrite_call(rewriter, code)
  assert ".astype" in res
  assert "jax.numpy.int64" in res


def test_metadata_missing_fallback(rewriter):
  """Verifies the behavior of metadata missing fallback."""
  cast_bad_def = {"variants": {"jax": {"api": "astype", "requires_plugin": "type_methods"}}}
  rewriter.semantics.get_definition_by_id.side_effect = lambda oid: cast_bad_def if oid == "CastBad" else None
  rewriter.ctx.current_op_id = "CastBad"
  call_node = cst.parse_expression("x.bad()")
  res_node = transform_casting(call_node, rewriter.ctx)
  assert res_node == call_node


def test_type_resolution_failure(rewriter):
  """Verifies the behavior of type resolution successfully handling failure."""
  cast_huge = {"metadata": {"target_type": "Int128"}, "variants": {"jax": {"requires_plugin": "type_methods"}}}
  rewriter.semantics.get_definition_by_id.side_effect = lambda oid: cast_huge if oid == "CastHuge" else None
  rewriter.ctx.current_op_id = "CastHuge"
  rewriter.semantics.resolve_variant.side_effect = lambda aid, fw: None
  call_node = cst.parse_expression("x.huge()")
  res_node = transform_casting(call_node, rewriter.ctx)
  assert res_node == call_node


def test_missing_semantics():
  """Verifies the behavior of missing semantics."""
  ctx = MagicMock()
  ctx.semantics = None
  node = cst.parse_expression("x.float()")
  res = transform_casting(node, ctx)
  assert res is node


def test_missing_conf(rewriter):
  """Verifies the behavior of missing conf."""
  rewriter.ctx.semantics.get_framework_config.return_value = None
  node = cst.parse_expression("x.float()")
  res = transform_casting(node, rewriter.ctx)
  assert res is node


def test_missing_traits(rewriter):
  """Verifies the behavior of missing traits."""
  rewriter.ctx.semantics.get_framework_config.return_value = {}
  node = cst.parse_expression("x.float()")
  res = transform_casting(node, rewriter.ctx)
  assert res is node


class MockTraits:
  """Mock Traits class for testing purposes."""

  def __init__(self, val):
    """Initializes the MockTraits instance."""
    self.has_numpy_compatible_arrays = val


def test_object_traits(rewriter):
  """Verifies the behavior of object traits."""
  rewriter.ctx.semantics.get_framework_config.return_value = {"plugin_traits": MockTraits(True)}
  rewriter.ctx.current_op_id = "CastFloat"
  node = cst.parse_expression("x.float()")
  res = transform_casting(node, rewriter.ctx)
  assert "astype" in cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res)])]).code


def test_object_traits_false(rewriter):
  """Verifies the behavior of object traits false."""
  rewriter.ctx.semantics.get_framework_config.return_value = {"plugin_traits": MockTraits(False)}
  rewriter.ctx.current_op_id = "CastFloat"
  node = cst.parse_expression("x.float()")
  res = transform_casting(node, rewriter.ctx)
  assert res is node


def test_object_traits_missing_attr(rewriter):
  """Verifies behavior when traits object lacks the attribute."""

  class EmptyTraits:
    """Empty traits."""

    pass

  rewriter.ctx.semantics.get_framework_config.return_value = {"plugin_traits": EmptyTraits()}
  rewriter.ctx.current_op_id = "CastFloat"
  node = cst.parse_expression("x.float()")
  res = transform_casting(node, rewriter.ctx)
  assert res is node


def test_non_attribute_call(rewriter):
  """Verifies the behavior of non attribute call."""
  rewriter.ctx.current_op_id = "CastFloat"
  node = cst.parse_expression("float(x)")
  res = transform_casting(node, rewriter.ctx)
  assert res is node


def test_missing_op_id(rewriter):
  """Verifies the behavior of missing op id."""
  rewriter.ctx.current_op_id = None
  node = cst.parse_expression("x.float()")
  res = transform_casting(node, rewriter.ctx)
  assert res is node


def test_missing_defn(rewriter):
  """Verifies the behavior of missing defn."""
  rewriter.ctx.current_op_id = "UnknownOp"
  rewriter.ctx.semantics.get_definition_by_id.return_value = None
  node = cst.parse_expression("x.float()")
  res = transform_casting(node, rewriter.ctx)
  assert res is node


def test_fallback_infer_type(rewriter):
  """Verifies the behavior of fallback infer type."""
  cast_half_def = {"variants": {}}

  def get_def_by_id(op_id):
    """Gets def by id."""
    if op_id == "CastHalf":
      return cast_half_def
    return None

  rewriter.ctx.semantics.get_definition_by_id.side_effect = get_def_by_id

  def resolve(aid, fw):
    """Resolves ."""
    if aid == "Float16" and fw == "jax":
      return {"api": "jax.numpy.float16"}
    return None

  rewriter.ctx.semantics.resolve_variant.side_effect = resolve
  rewriter.ctx.current_op_id = "CastHalf"
  node = cst.parse_expression("x.half()")
  res = transform_casting(node, rewriter.ctx)
  res_code = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res)])]).code
  assert "astype" in res_code
  assert "jax.numpy.float16" in res_code


def test_fallback_infer_type_unmapped(rewriter):
  """Verifies the behavior of fallback infer type unmapped."""
  cast_unknown_def = {"variants": {}}

  def get_def_by_id(op_id):
    """Gets def by id."""
    if op_id == "CastUnknown":
      return cast_unknown_def
    return None

  rewriter.ctx.semantics.get_definition_by_id.side_effect = get_def_by_id
  rewriter.ctx.current_op_id = "CastUnknown"
  node = cst.parse_expression("x.unknown()")
  res = transform_casting(node, rewriter.ctx)
  assert res is node
