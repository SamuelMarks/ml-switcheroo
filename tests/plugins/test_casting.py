"""
Tests for Type Casting Plugin.

Verifies:
1. Retrieval of Abstract Type ID from Operation Metadata.
2. Lookup of Target Framework API for that Type.
3. Generation of `.astype()` calls.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

# Fix: Import TestRewriter shim
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.casting import transform_casting


def rewrite_call(rewriter, code):
  """Executes specific rewriter phase on code string."""
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter():
  """Function docstring."""
  # Register hook manually
  hooks._HOOKS["type_methods"] = transform_casting
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  # --- 1. Define Abstract Cast Operations (with Metadata) ---
  cast_float_def = {
    "variants": {
      "torch": {"api": "torch.Tensor.float"},
      "jax": {"api": "astype", "requires_plugin": "type_methods"},
    },
    "metadata": {"target_type": "Float32"},  # Metadata Link
  }

  cast_long_def = {
    "variants": {
      "torch": {"api": "torch.Tensor.long"},
      "jax": {"api": "astype", "requires_plugin": "type_methods"},
    },
    "metadata": {"target_type": "Int64"},  # Metadata Link
  }

  # --- 2. Define Abstract Types (Target mapping) ---
  float32_def = {"variants": {"jax": {"api": "jax.numpy.float32"}}}
  int64_def = {"variants": {"jax": {"api": "jax.numpy.int64"}}}

  # Aggregate Definitions
  all_defs = {
    "CastFloat": cast_float_def,
    "CastLong": cast_long_def,
    "Float32": float32_def,
    "Int64": int64_def,
  }

  # Mock Lookup Logic
  def get_def(name):
    """Function docstring."""
    if "float" in name:
      return ("CastFloat", cast_float_def)
    if "long" in name:
      return ("CastLong", cast_long_def)
    return None

  def get_def_by_id(op_id):
    """Function docstring."""
    return all_defs.get(op_id)

  def resolve(aid, fw):
    """Function docstring."""
    defn = all_defs.get(aid)
    if defn and fw in defn["variants"]:
      return defn["variants"][fw]
    return None

  # Wire up mocks
  mgr.get_definition.side_effect = get_def
  mgr.get_definition_by_id.side_effect = get_def_by_id
  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = all_defs
  mgr.is_verified.return_value = True

  # Safe Config Defaults
  mgr.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": True}}

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_float_cast(rewriter):
  """
  Input: x.float()
  Logic:
    1. Map .float() -> 'CastFloat'
    2. 'CastFloat' metadata -> 'Float32'
    3. Resolve 'Float32' (JAX) -> 'jax.numpy.float32'
  Output: x.astype(jax.numpy.float32)
  """
  # Rewriter requires context propagation
  rewriter.ctx.current_op_id = "CastFloat"

  code = "y = x.float()"
  res = rewrite_call(rewriter, code)

  assert ".astype" in res
  assert "jax.numpy.float32" in res


def test_long_cast(rewriter):
  """
  Input: x.long()
  Logic: 'CastLong' -> 'Int64' -> 'jax.numpy.int64'
  """
  rewriter.ctx.current_op_id = "CastLong"

  code = "idx = mask.long()"
  res = rewrite_call(rewriter, code)

  assert ".astype" in res
  assert "jax.numpy.int64" in res


def test_metadata_missing_fallback(rewriter):
  """
  Scenario: Op definition exists but metadata missing.
  Expectation: Return original node.
  """
  # Inject bad definition
  cast_bad_def = {
    "variants": {"jax": {"api": "astype", "requires_plugin": "type_methods"}},
    # Missing metadata
  }
  rewriter.semantics.get_definition_by_id.side_effect = lambda oid: cast_bad_def if oid == "CastBad" else None

  rewriter.ctx.current_op_id = "CastBad"

  call_node = cst.parse_expression("x.bad()")
  res_node = transform_casting(call_node, rewriter.ctx)

  # Should remain unchanged
  assert res_node == call_node


def test_type_resolution_failure(rewriter):
  """
  Scenario: Target Framework doesn't map the abstract type.
  Expectation: Return original node.
  """
  # Define cast asking for 'Int128'
  cast_huge = {
    "metadata": {"target_type": "Int128"},
    "variants": {"jax": {"requires_plugin": "type_methods"}},
  }
  rewriter.semantics.get_definition_by_id.side_effect = lambda oid: cast_huge if oid == "CastHuge" else None
  rewriter.ctx.current_op_id = "CastHuge"

  # Ensure Lookup fails for Int128
  rewriter.semantics.resolve_variant.side_effect = lambda aid, fw: None

  call_node = cst.parse_expression("x.huge()")
  res_node = transform_casting(call_node, rewriter.ctx)

  assert res_node == call_node


# --- Extra Coverage Tests ---
from ml_switcheroo.core.hooks import HookContext


def test_missing_semantics():
  ctx = MagicMock()
  ctx.semantics = None
  # No semantics attached
  node = cst.parse_expression("x.float()")
  res = transform_casting(node, ctx)
  assert res is node


def test_missing_conf(rewriter):
  rewriter.ctx.semantics.get_framework_config.return_value = None
  node = cst.parse_expression("x.float()")
  res = transform_casting(node, rewriter.ctx)
  assert res is node


def test_missing_traits(rewriter):
  rewriter.ctx.semantics.get_framework_config.return_value = {}
  node = cst.parse_expression("x.float()")
  res = transform_casting(node, rewriter.ctx)
  assert res is node


class MockTraits:
  def __init__(self, val):
    self.has_numpy_compatible_arrays = val


def test_object_traits(rewriter):
  rewriter.ctx.semantics.get_framework_config.return_value = {"plugin_traits": MockTraits(True)}
  rewriter.ctx.current_op_id = "CastFloat"
  node = cst.parse_expression("x.float()")
  res = transform_casting(node, rewriter.ctx)
  assert "astype" in cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res)])]).code


def test_object_traits_false(rewriter):
  rewriter.ctx.semantics.get_framework_config.return_value = {"plugin_traits": MockTraits(False)}
  rewriter.ctx.current_op_id = "CastFloat"
  node = cst.parse_expression("x.float()")
  res = transform_casting(node, rewriter.ctx)
  assert res is node


def test_non_attribute_call(rewriter):
  # node.func is not an attribute
  rewriter.ctx.current_op_id = "CastFloat"
  node = cst.parse_expression("float(x)")
  res = transform_casting(node, rewriter.ctx)
  assert res is node


def test_missing_op_id(rewriter):
  rewriter.ctx.current_op_id = None
  node = cst.parse_expression("x.float()")
  res = transform_casting(node, rewriter.ctx)
  assert res is node


def test_missing_defn(rewriter):
  rewriter.ctx.current_op_id = "UnknownOp"
  rewriter.ctx.semantics.get_definition_by_id.return_value = None
  node = cst.parse_expression("x.float()")
  res = transform_casting(node, rewriter.ctx)
  assert res is node


def test_fallback_infer_type(rewriter):
  # Fallback: Infer type from Op ID if metadata missing
  # Op ID: CastHalf -> Float16
  cast_half_def = {"variants": {}}

  def get_def_by_id(op_id):
    if op_id == "CastHalf":
      return cast_half_def
    return None

  rewriter.ctx.semantics.get_definition_by_id.side_effect = get_def_by_id

  # Resolve Float16 -> 'jax.numpy.float16'
  def resolve(aid, fw):
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
  # Op ID: CastUnknown -> Unknown
  cast_unknown_def = {"variants": {}}

  def get_def_by_id(op_id):
    if op_id == "CastUnknown":
      return cast_unknown_def
    return None

  rewriter.ctx.semantics.get_definition_by_id.side_effect = get_def_by_id

  rewriter.ctx.current_op_id = "CastUnknown"
  node = cst.parse_expression("x.unknown()")
  res = transform_casting(node, rewriter.ctx)
  assert res is node
