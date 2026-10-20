"""Test suite for the Casting module."""

import typing
from pydantic import JsonValue
from unittest.mock import MagicMock

import libcst as cst
import pytest

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.casting import _supports_numpy_casting, transform_casting
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_call(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites call.

  Args:
      rewriter: Test argument or fixture.
      code: Test argument or fixture.

  Returns:
      Computed test result or fixture value.
  """
  return str(rewriter.convert(cst.parse_module(code)).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Execute rewriter helper logic.

  Returns:
      Computed test result or fixture value.
  """
  hooks._HOOKS["type_methods"] = transform_casting  # type: ignore[assignment]
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
  all_defs = {
    "CastFloat": cast_float_def,
    "CastLong": cast_long_def,
    "Float32": float32_def,
    "Int64": int64_def,
  }

  def get_def(name: str) -> typing.Optional[tuple[str, dict[str, JsonValue]]]:
    """Gets def.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if "float" in name:
      return ("CastFloat", cast_float_def)  # type: ignore[return-value]
    if "long" in name:
      return ("CastLong", cast_long_def)  # type: ignore[return-value]
    return None

  def get_def_by_id(op_id: str) -> typing.Optional[dict[str, JsonValue]]:
    """Gets def by id.

    Args:
        op_id: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return all_defs.get(op_id)  # type: ignore[return-value]

  def resolve(aid: str, fw: str) -> typing.Optional[dict[str, JsonValue]]:
    """Resolves .

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    defn = all_defs.get(aid)
    if defn and fw in defn["variants"]:  # type: ignore[index]
      res = defn["variants"][fw]  # type: ignore[index]
      return res if isinstance(res, dict) else {}
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
  """Verifies the behavior of float cast.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.ctx.current_op_id = "CastFloat"
  code: str = "y = x.float()"
  res: str = rewrite_call(rewriter, code)
  assert ".astype" in res
  assert "jax.numpy.float32" in res


def test_long_cast(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of long cast.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.ctx.current_op_id = "CastLong"
  code: str = "idx = mask.long()"
  res: str = rewrite_call(rewriter, code)
  assert ".astype" in res
  assert "jax.numpy.int64" in res


def test_metadata_missing_fallback(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of metadata missing fallback.

  Args:
      rewriter: Test argument or fixture.
  """
  cast_bad_def = {"variants": {"jax": {"api": "astype", "requires_plugin": "type_methods"}}}
  rewriter.semantics.get_definition_by_id.side_effect = lambda oid: cast_bad_def if oid == "CastBad" else None  # type: ignore[attr-defined]
  rewriter.ctx.current_op_id = "CastBad"
  call_node: cst.Call = cst.ensure_type(cst.parse_expression("x.bad()"), cst.Call)
  res_node: cst.CSTNode = transform_casting(call_node, rewriter.ctx)
  assert res_node == call_node


def test_type_resolution_failure(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of type resolution successfully handling failure.

  Args:
      rewriter: Test argument or fixture.
  """
  cast_huge = {
    "metadata": {"target_type": "Int128"},
    "variants": {"jax": {"requires_plugin": "type_methods"}},
  }
  rewriter.semantics.get_definition_by_id.side_effect = lambda oid: cast_huge if oid == "CastHuge" else None  # type: ignore[attr-defined]
  rewriter.ctx.current_op_id = "CastHuge"
  rewriter.semantics.resolve_variant.side_effect = lambda aid, fw: None  # type: ignore[attr-defined]
  call_node: cst.Call = cst.ensure_type(cst.parse_expression("x.huge()"), cst.Call)
  res_node: cst.CSTNode = transform_casting(call_node, rewriter.ctx)
  assert res_node == call_node


def test_missing_semantics() -> None:
  """Verifies the behavior of missing semantics."""
  ctx = MagicMock()
  ctx.semantics = None
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, ctx)
  assert res is node


def test_missing_conf(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of missing conf.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.ctx.semantics.get_framework_config.return_value = None  # type: ignore[union-attr]
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_missing_traits(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of missing traits.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.ctx.semantics.get_framework_config.return_value = {}  # type: ignore[union-attr]
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


class MockTraits:
  """Test helper class MockTraits."""

  def __init__(self, val: bool) -> None:
    """Initializes the MockTraits instance.

    Args:
        val: Test argument or fixture.
    """
    self.has_numpy_compatible_arrays: bool = val


def test_object_traits(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of object traits.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.ctx.semantics.get_framework_config.return_value = {"plugin_traits": MockTraits(True)}  # type: ignore[union-attr]
  rewriter.ctx.current_op_id = "CastFloat"
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert "astype" in cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res, cst.Call))])]).code


def test_object_traits_false(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of object traits false.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.ctx.semantics.get_framework_config.return_value = {"plugin_traits": MockTraits(False)}  # type: ignore[union-attr]
  rewriter.ctx.current_op_id = "CastFloat"
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_object_traits_missing_attr(rewriter: PivotRewriter) -> None:
  """Verifies behavior when traits object lacks the attribute.

  Args:
      rewriter: Test argument or fixture.
  """

  class EmptyTraits:
    """Empty traits."""

    pass

  rewriter.ctx.semantics.get_framework_config.return_value = {"plugin_traits": EmptyTraits()}  # type: ignore[union-attr]
  rewriter.ctx.current_op_id = "CastFloat"
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_non_attribute_call(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of non attribute call.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.ctx.current_op_id = "CastFloat"
  node: cst.Call = cst.ensure_type(cst.parse_expression("float(x)"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_missing_op_id(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of missing op id.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.ctx.current_op_id = None
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_missing_defn(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of missing defn.

  Args:
      rewriter: Test argument or fixture.
  """
  rewriter.ctx.current_op_id = "UnknownOp"
  rewriter.ctx.semantics.get_definition_by_id.return_value = None  # type: ignore[union-attr]
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_fallback_infer_type(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of fallback infer type.

  Args:
      rewriter: Test argument or fixture.
  """
  cast_half_def: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {"variants": {}}

  def get_def_by_id(op_id: str) -> typing.Optional[dict[str, JsonValue]]:
    """Gets def by id.

    Args:
        op_id: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if op_id == "CastHalf":
      return cast_half_def  # type: ignore[return-value]
    return None

  rewriter.ctx.semantics.get_definition_by_id.side_effect = get_def_by_id  # type: ignore[union-attr]

  def resolve(aid: str, fw: str) -> typing.Optional[dict[str, JsonValue]]:
    """Resolves .

    Args:
        aid: Test argument or fixture.
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if aid == "Float16" and fw == "jax":
      return {"api": "jax.numpy.float16"}
    return None

  rewriter.ctx.semantics.resolve_variant.side_effect = resolve  # type: ignore[union-attr]
  rewriter.ctx.current_op_id = "CastHalf"
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.half()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  res_code: str = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res, cst.Call))])]).code
  assert "astype" in res_code
  assert "jax.numpy.float16" in res_code


def test_fallback_infer_type_unmapped(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of fallback infer type unmapped.

  Args:
      rewriter: Test argument or fixture.
  """
  cast_unknown_def: dict[str, typing.Union[typing.Hashable, typing.Sized]] = {"variants": {}}

  def get_def_by_id(op_id: str) -> typing.Optional[dict[str, JsonValue]]:
    """Gets def by id.

    Args:
        op_id: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if op_id == "CastUnknown":
      return cast_unknown_def  # type: ignore[return-value]
    return None

  rewriter.ctx.semantics.get_definition_by_id.side_effect = get_def_by_id  # type: ignore[union-attr]
  rewriter.ctx.current_op_id = "CastUnknown"
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.unknown()"), cst.Call)
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
  result = transform_casting(node, ctx)
  assert result is node
