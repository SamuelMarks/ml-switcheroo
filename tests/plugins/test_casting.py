"""Test suite for the Casting module."""

import typing
from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.casting import _supports_numpy_casting, transform_casting
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_call(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites call."""
  return typing.cast(str, rewriter.convert(cst.parse_module(code)).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Docstring."""
  hooks._HOOKS["type_methods"] = transform_casting
  hooks._PLUGINS_LOADED = True
  mgr = MagicMock()
  cast_float_def: dict[str, typing.Any] = {
    "variants": {"torch": {"api": "torch.Tensor.float"}, "jax": {"api": "astype", "requires_plugin": "type_methods"}},
    "metadata": {"target_type": "Float32"},
  }
  cast_long_def: dict[str, typing.Any] = {
    "variants": {"torch": {"api": "torch.Tensor.long"}, "jax": {"api": "astype", "requires_plugin": "type_methods"}},
    "metadata": {"target_type": "Int64"},
  }
  float32_def: dict[str, typing.Any] = {"variants": {"jax": {"api": "jax.numpy.float32"}}}
  int64_def: dict[str, typing.Any] = {"variants": {"jax": {"api": "jax.numpy.int64"}}}
  all_defs: dict[str, dict[str, typing.Any]] = {
    "CastFloat": cast_float_def,
    "CastLong": cast_long_def,
    "Float32": float32_def,
    "Int64": int64_def,
  }

  def get_def(name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Gets def."""
    if "float" in name:
      return ("CastFloat", cast_float_def)
    if "long" in name:
      return ("CastLong", cast_long_def)
    return None

  def get_def_by_id(op_id: str) -> typing.Optional[dict[str, typing.Any]]:
    """Gets def by id."""
    return all_defs.get(op_id)

  def resolve(aid: str, fw: str) -> typing.Optional[dict[str, typing.Any]]:
    """Resolves ."""
    defn = all_defs.get(aid)
    if defn and fw in defn["variants"]:
      return typing.cast(dict[str, typing.Any], defn["variants"][fw])
    return None

  mgr.get_definition.side_effect = get_def
  mgr.get_definition_by_id.side_effect = get_def_by_id
  mgr.resolve_variant.side_effect = resolve
  mgr.get_known_apis.return_value = all_defs
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": True}}
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_float_cast(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of float cast."""
  rewriter.ctx.current_op_id = "CastFloat"
  code: str = "y = x.float()"
  res: str = rewrite_call(rewriter, code)
  assert ".astype" in res
  assert "jax.numpy.float32" in res


def test_long_cast(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of long cast."""
  rewriter.ctx.current_op_id = "CastLong"
  code: str = "idx = mask.long()"
  res: str = rewrite_call(rewriter, code)
  assert ".astype" in res
  assert "jax.numpy.int64" in res


def test_metadata_missing_fallback(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of metadata missing fallback."""
  cast_bad_def: dict[str, typing.Any] = {"variants": {"jax": {"api": "astype", "requires_plugin": "type_methods"}}}
  rewriter.semantics.get_definition_by_id.side_effect = lambda oid: cast_bad_def if oid == "CastBad" else None
  rewriter.ctx.current_op_id = "CastBad"
  call_node: cst.BaseExpression = cst.parse_expression("x.bad()")
  res_node: cst.CSTNode = transform_casting(call_node, rewriter.ctx)
  assert res_node == call_node


def test_type_resolution_failure(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of type resolution successfully handling failure."""
  cast_huge: dict[str, typing.Any] = {
    "metadata": {"target_type": "Int128"},
    "variants": {"jax": {"requires_plugin": "type_methods"}},
  }
  rewriter.semantics.get_definition_by_id.side_effect = lambda oid: cast_huge if oid == "CastHuge" else None
  rewriter.ctx.current_op_id = "CastHuge"
  rewriter.semantics.resolve_variant.side_effect = lambda aid, fw: None
  call_node: cst.BaseExpression = cst.parse_expression("x.huge()")
  res_node: cst.CSTNode = transform_casting(call_node, rewriter.ctx)
  assert res_node == call_node


def test_missing_semantics() -> None:
  """Verifies the behavior of missing semantics."""
  ctx = MagicMock()
  ctx.semantics = None
  node: cst.BaseExpression = cst.parse_expression("x.float()")
  res: cst.CSTNode = transform_casting(node, ctx)
  assert res is node


def test_missing_conf(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of missing conf."""
  rewriter.ctx.semantics.get_framework_config.return_value = None
  node: cst.BaseExpression = cst.parse_expression("x.float()")
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_missing_traits(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of missing traits."""
  rewriter.ctx.semantics.get_framework_config.return_value = {}
  node: cst.BaseExpression = cst.parse_expression("x.float()")
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


class MockTraits:
  """Docstring."""

  def __init__(self, val: bool) -> None:
    """Initializes the MockTraits instance."""
    self.has_numpy_compatible_arrays: bool = val


def test_object_traits(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of object traits."""
  rewriter.ctx.semantics.get_framework_config.return_value = {"plugin_traits": MockTraits(True)}
  rewriter.ctx.current_op_id = "CastFloat"
  node: cst.BaseExpression = cst.parse_expression("x.float()")
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert "astype" in cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res))])]).code


def test_object_traits_false(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of object traits false."""
  rewriter.ctx.semantics.get_framework_config.return_value = {"plugin_traits": MockTraits(False)}
  rewriter.ctx.current_op_id = "CastFloat"
  node: cst.BaseExpression = cst.parse_expression("x.float()")
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_object_traits_missing_attr(rewriter: PivotRewriter) -> None:
  """Verifies behavior when traits object lacks the attribute."""

  class EmptyTraits:
    """Empty traits."""

    pass

  rewriter.ctx.semantics.get_framework_config.return_value = {"plugin_traits": EmptyTraits()}
  rewriter.ctx.current_op_id = "CastFloat"
  node: cst.BaseExpression = cst.parse_expression("x.float()")
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_non_attribute_call(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of non attribute call."""
  rewriter.ctx.current_op_id = "CastFloat"
  node: cst.BaseExpression = cst.parse_expression("float(x)")
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_missing_op_id(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of missing op id."""
  rewriter.ctx.current_op_id = None
  node: cst.BaseExpression = cst.parse_expression("x.float()")
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_missing_defn(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of missing defn."""
  rewriter.ctx.current_op_id = "UnknownOp"
  rewriter.ctx.semantics.get_definition_by_id.return_value = None
  node: cst.BaseExpression = cst.parse_expression("x.float()")
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_fallback_infer_type(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of fallback infer type."""
  cast_half_def: dict[str, typing.Any] = {"variants": {}}

  def get_def_by_id(op_id: str) -> typing.Optional[dict[str, typing.Any]]:
    """Gets def by id."""
    if op_id == "CastHalf":
      return cast_half_def
    return None

  rewriter.ctx.semantics.get_definition_by_id.side_effect = get_def_by_id

  def resolve(aid: str, fw: str) -> typing.Optional[dict[str, typing.Any]]:
    """Resolves ."""
    if aid == "Float16" and fw == "jax":
      return {"api": "jax.numpy.float16"}
    return None

  rewriter.ctx.semantics.resolve_variant.side_effect = resolve
  rewriter.ctx.current_op_id = "CastHalf"
  node: cst.BaseExpression = cst.parse_expression("x.half()")
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  res_code: str = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(typing.cast(cst.BaseExpression, res))])]).code
  assert "astype" in res_code
  assert "jax.numpy.float16" in res_code


def test_fallback_infer_type_unmapped(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of fallback infer type unmapped."""
  cast_unknown_def: dict[str, typing.Any] = {"variants": {}}

  def get_def_by_id(op_id: str) -> typing.Optional[dict[str, typing.Any]]:
    """Gets def by id."""
    if op_id == "CastUnknown":
      return cast_unknown_def
    return None

  rewriter.ctx.semantics.get_definition_by_id.side_effect = get_def_by_id
  rewriter.ctx.current_op_id = "CastUnknown"
  node: cst.BaseExpression = cst.parse_expression("x.unknown()")
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


# --- Merged from test_casting_extra.py ---


def test_casting_missing_traits_in_conf() -> None:
  """Verifies the behavior of casting missing traits in conf."""
  semantics = MagicMock()
  ctx = HookContext(semantics=semantics, config=MagicMock(effective_target="jax"))
  semantics.get_framework_config.return_value = {"plugin_traits": None}
  assert _supports_numpy_casting(ctx) is False


def test_casting_op_id_not_cast() -> None:
  """Verifies the behavior of casting op id not cast."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("float")))
  semantics = MagicMock()
  ctx = HookContext(semantics=semantics, config=MagicMock(effective_target="jax"))
  semantics.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": True}}
  ctx.current_op_id = "SomethingElse"
  semantics.get_definition_by_id.return_value = {"metadata": {}}
  result: typing.Any = transform_casting(node, ctx)
  assert result is node
