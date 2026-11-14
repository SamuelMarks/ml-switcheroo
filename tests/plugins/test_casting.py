"""Test suite for the Casting module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest
from pydantic import JsonValue

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.casting import _supports_numpy_casting, transform_casting
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_call(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites call.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      code: Python or DSL source code string to parse or transpile.

  Returns:
      Resulting `str` from rewrite call.
  """
  return str(rewriter.convert(cst.parse_module(code)).code)


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Create a configured `PivotRewriter` fixture for casting tests.

  Returns:
      Configured `rewriter` fixture value (PivotRewriter).
  """
  hooks.register_hook("type_methods")(transform_casting)
  hooks.set_plugins_loaded(True)
  mgr = MagicMock()
  cast_float_def: dict[str, JsonValue] = {
    "variants": {"torch": {"api": "torch.Tensor.float"}, "jax": {"api": "astype", "requires_plugin": "type_methods"}},
    "metadata": {"target_type": "Float32"},
  }
  cast_long_def: dict[str, JsonValue] = {
    "variants": {"torch": {"api": "torch.Tensor.long"}, "jax": {"api": "astype", "requires_plugin": "type_methods"}},
    "metadata": {"target_type": "Int64"},
  }
  float32_def: dict[str, JsonValue] = {"variants": {"jax": {"api": "jax.numpy.float32"}}}
  int64_def: dict[str, JsonValue] = {"variants": {"jax": {"api": "jax.numpy.int64"}}}
  all_defs: dict[str, dict[str, JsonValue]] = {
    "CastFloat": cast_float_def,
    "CastLong": cast_long_def,
    "Float32": float32_def,
    "Int64": int64_def,
  }

  def get_def(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get def.
    """
    if "float" in name:
      return ("CastFloat", cast_float_def)
    if "long" in name:
      return ("CastLong", cast_long_def)
    return None

  def get_def_by_id(op_id: str) -> dict[str, JsonValue] | None:
    """Gets def by id.

    Args:
        op_id: Input `op_id` parameter value (`str`).

    Returns:
        Resulting `dict[str, JsonValue] | None` from get def by id.
    """
    return all_defs.get(op_id)

  def resolve(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Execute `all_defs.get` to support resolve in casting tests.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve.
    """
    defn = all_defs.get(aid)
    if defn is not None:
      variants = defn.get("variants")
      if isinstance(variants, dict) and fw in variants:
        res = variants[fw]
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
  """Verify `rewrite_call` emits `.astype` for float cast.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  rewriter.ctx.current_op_id = "CastFloat"
  code: str = "y = x.float()"
  res: str = rewrite_call(rewriter, code)
  assert ".astype" in res
  assert "jax.numpy.float32" in res


def test_long_cast(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_call` emits `.astype` for long cast.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  rewriter.ctx.current_op_id = "CastLong"
  code: str = "idx = mask.long()"
  res: str = rewrite_call(rewriter, code)
  assert ".astype" in res
  assert "jax.numpy.int64" in res


def test_metadata_missing_fallback(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ensure_type` handles metadata missing fallback as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  cast_bad_def = {"variants": {"jax": {"api": "astype", "requires_plugin": "type_methods"}}}
  monkeypatch.setattr(
    rewriter.semantics.get_definition_by_id,
    "side_effect",
    (lambda oid: cast_bad_def if oid == "CastBad" else None),
    raising=False,
  )
  rewriter.ctx.current_op_id = "CastBad"
  call_node: cst.Call = cst.ensure_type(cst.parse_expression("x.bad()"), cst.Call)
  res_node: cst.CSTNode = transform_casting(call_node, rewriter.ctx)
  assert res_node == call_node


def test_type_resolution_failure(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ensure_type` handles type resolution failure as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  cast_huge = {
    "metadata": {"target_type": "Int128"},
    "variants": {"jax": {"requires_plugin": "type_methods"}},
  }
  monkeypatch.setattr(
    rewriter.semantics.get_definition_by_id,
    "side_effect",
    (lambda oid: cast_huge if oid == "CastHuge" else None),
    raising=False,
  )
  rewriter.ctx.current_op_id = "CastHuge"
  monkeypatch.setattr(
    rewriter.semantics.resolve_variant,
    "side_effect",
    (lambda aid, fw: None),
    raising=False,
  )
  call_node: cst.Call = cst.ensure_type(cst.parse_expression("x.huge()"), cst.Call)
  res_node: cst.CSTNode = transform_casting(call_node, rewriter.ctx)
  assert res_node == call_node


def test_missing_semantics() -> None:
  """Verify `ensure_type` handles missing semantics as expected."""
  ctx = MagicMock()
  ctx.semantics = None
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, ctx)
  assert res is node


def test_missing_conf(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ensure_type` handles missing conf as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  assert rewriter.ctx.semantics is not None
  monkeypatch.setattr(
    rewriter.ctx.semantics.get_framework_config,
    "return_value",
    None,
    raising=False,
  )
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_missing_traits(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ensure_type` handles missing traits as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  assert rewriter.ctx.semantics is not None
  monkeypatch.setattr(
    rewriter.ctx.semantics.get_framework_config,
    "return_value",
    {},
    raising=False,
  )
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


class MockTraits:
  """Sentinel `MockTraits` fixture class used in casting tests."""

  def __init__(self, val: bool) -> None:
    """Initializes the MockTraits instance.

    Args:
        val: Input `val` parameter value (`bool`).
    """
    self.has_numpy_compatible_arrays: bool = val


def test_object_traits(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ensure_type` emits `astype` for object traits.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  assert rewriter.ctx.semantics is not None
  monkeypatch.setattr(
    rewriter.ctx.semantics.get_framework_config,
    "return_value",
    {"plugin_traits": MockTraits(True)},
    raising=False,
  )
  rewriter.ctx.current_op_id = "CastFloat"
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert "astype" in cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res, cst.Call))])]).code


def test_object_traits_false(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ensure_type` handles object traits false as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  assert rewriter.ctx.semantics is not None
  monkeypatch.setattr(
    rewriter.ctx.semantics.get_framework_config,
    "return_value",
    {"plugin_traits": MockTraits(False)},
    raising=False,
  )
  rewriter.ctx.current_op_id = "CastFloat"
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_object_traits_missing_attr(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies behavior when traits object lacks the attribute.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """

  class EmptyTraits:
    """Empty traits."""

  assert rewriter.ctx.semantics is not None
  monkeypatch.setattr(
    rewriter.ctx.semantics.get_framework_config,
    "return_value",
    {"plugin_traits": EmptyTraits()},
    raising=False,
  )
  rewriter.ctx.current_op_id = "CastFloat"
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_non_attribute_call(rewriter: PivotRewriter) -> None:
  """Verify `ensure_type` handles non attribute call as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  rewriter.ctx.current_op_id = "CastFloat"
  node: cst.Call = cst.ensure_type(cst.parse_expression("float(x)"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_missing_op_id(rewriter: PivotRewriter) -> None:
  """Verify `ensure_type` handles missing operation id as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
  """
  rewriter.ctx.current_op_id = None
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_missing_defn(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ensure_type` handles missing defn as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  rewriter.ctx.current_op_id = "UnknownOp"
  assert rewriter.ctx.semantics is not None
  monkeypatch.setattr(
    rewriter.ctx.semantics.get_definition_by_id,
    "return_value",
    None,
    raising=False,
  )
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.float()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_fallback_infer_type(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ensure_type` emits `astype` for fallback infer type.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  cast_half_def: dict[str, JsonValue] = {"variants": {}}

  def get_def_by_id(op_id: str) -> dict[str, JsonValue] | None:
    """Gets def by id.

    Args:
        op_id: Input `op_id` parameter value (`str`).

    Returns:
        Resulting `dict[str, JsonValue] | None` from get def by id.
    """
    if op_id == "CastHalf":
      return cast_half_def
    return None

  assert rewriter.ctx.semantics is not None
  monkeypatch.setattr(
    rewriter.ctx.semantics.get_definition_by_id,
    "side_effect",
    get_def_by_id,
    raising=False,
  )

  def resolve(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Execute `resolve` helper in casting tests.

    Args:
        aid: Abstract operation identifier string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve.
    """
    if aid == "Float16" and fw == "jax":
      return {"api": "jax.numpy.float16"}
    return None

  monkeypatch.setattr(
    rewriter.ctx.semantics.resolve_variant,
    "side_effect",
    resolve,
    raising=False,
  )
  rewriter.ctx.current_op_id = "CastHalf"
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.half()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  res_code: str = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(cst.ensure_type(res, cst.Call))])]).code
  assert "astype" in res_code
  assert "jax.numpy.float16" in res_code


def test_fallback_infer_type_unmapped(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify `ensure_type` handles fallback infer type unmapped as expected.

  Args:
      rewriter: Configured CST rewriter test helper instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  cast_unknown_def: dict[str, JsonValue] = {"variants": {}}

  def get_def_by_id(op_id: str) -> dict[str, JsonValue] | None:
    """Gets def by id.

    Args:
        op_id: Input `op_id` parameter value (`str`).

    Returns:
        Resulting `dict[str, JsonValue] | None` from get def by id.
    """
    if op_id == "CastUnknown":
      return cast_unknown_def
    return None

  assert rewriter.ctx.semantics is not None
  monkeypatch.setattr(
    rewriter.ctx.semantics.get_definition_by_id,
    "side_effect",
    get_def_by_id,
    raising=False,
  )
  rewriter.ctx.current_op_id = "CastUnknown"
  node: cst.Call = cst.ensure_type(cst.parse_expression("x.unknown()"), cst.Call)
  res: cst.CSTNode = transform_casting(node, rewriter.ctx)
  assert res is node


def test_casting_missing_traits_in_conf() -> None:
  """Verify `HookContext` handles casting missing traits in conf as expected."""
  semantics = MagicMock()
  ctx = HookContext(semantics=semantics, config=MagicMock(effective_target="jax"))
  semantics.get_framework_config.return_value = {"plugin_traits": None}
  assert _supports_numpy_casting(ctx) is False


def test_casting_op_id_not_cast() -> None:
  """Verify `Call` handles casting operation id not cast as expected."""
  node = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("float")))
  semantics = MagicMock()
  ctx = HookContext(semantics=semantics, config=MagicMock(effective_target="jax"))
  semantics.get_framework_config.return_value = {"plugin_traits": {"has_numpy_compatible_arrays": True}}
  ctx.current_op_id = "SomethingElse"
  semantics.get_definition_by_id.return_value = {"metadata": {}}
  result = transform_casting(node, ctx)
  assert result is node
