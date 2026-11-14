"""Test suite for the Method Property module."""

from unittest.mock import MagicMock

import libcst as cst
import pytest
from pydantic import JsonValue

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.method_property import transform_method_to_property
from tests.conftest import TestRewriter as PivotRewriter


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      code (str): The code.

  Returns:
      str: The rewritten code string.
  """
  return rewriter.convert(cst.parse_module(code)).code


@pytest.fixture
def rewriter() -> PivotRewriter:
  """Provides a mock rewriter for testing.

  Returns:
      PivotRewriter: A mock rewriter instance.
  """
  hooks.register_hook("method_to_property")(transform_method_to_property)
  hooks.set_plugins_loaded(True)
  mgr: MagicMock = MagicMock()
  size_variants: dict[str, dict[str, JsonValue]] = {"jax": {"api": "shape", "requires_plugin": "method_to_property"}}
  data_ptr_variants: dict[str, dict[str, JsonValue]] = {"jax": {"api": "data", "requires_plugin": "method_to_property"}}
  size_def: dict[str, JsonValue] = {"variants": dict[str, JsonValue](size_variants)}
  data_ptr_def: dict[str, JsonValue] = {"variants": dict[str, JsonValue](data_ptr_variants)}
  all_defs: dict[str, dict[str, JsonValue]] = {"size": size_def, "data_ptr": data_ptr_def}
  all_variants: dict[str, dict[str, dict[str, JsonValue]]] = {"size": size_variants, "data_ptr": data_ptr_variants}

  def get_def_side_effect(name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Gets def side effect.

    Args:
        name (str): Definition name.

    Returns:
        Optional[Tuple[str, dict[str, JsonValue]]]: Definition or None.
    """
    if name == "size" or name.endswith(".size"):
      return ("size", size_def)
    return None

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.get_known_apis.return_value = all_defs

  def resolve_variant_side_effect(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Resolves variant side effect.

    Args:
        aid (str): Definition ID.
        fw (str): Target framework string.

    Returns:
        Optional[dict[str, JsonValue]]: Variant definition.
    """
    if aid in all_variants:
      return all_variants[aid].get(fw)
    return None

  mgr.resolve_variant.side_effect = resolve_variant_side_effect
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}
  cfg: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_simple_size_conversion(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `x.shape` for simple size conversion.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  assert "x.shape" in rewrite_code(rewriter, "s = x.size()")


def test_indexed_size_conversion(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `x.shape[0]` for indexed size conversion.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  assert "x.shape[0]" in rewrite_code(rewriter, "d = x.size(0)").replace(" ", "")


def test_ignore_other_methods(rewriter: PivotRewriter) -> None:
  """Verify `rewrite_code` emits `x.other()` for ignore other methods.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  assert "x.other()" in rewrite_code(rewriter, "x.other()")


def test_obj_type_not_tensor(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies that the method is not rewritten if the receiver is known to not be a tensor.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr(rewriter.ctx, "resolve_type", MagicMock(return_value="Module"), raising=False)
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("size")))
  res: cst.CSTNode | cst.Subscript | cst.Attribute = transform_method_to_property(node, rewriter.ctx)
  assert res is node


def test_missing_target_prop(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies that the method is not rewritten if lookup_api fails.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr(rewriter.ctx, "resolve_type", MagicMock(return_value="Tensor"), raising=False)
  monkeypatch.setattr(rewriter.ctx, "lookup_api", MagicMock(return_value=None), raising=False)
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("size")))
  res: cst.CSTNode | cst.Subscript | cst.Attribute = transform_method_to_property(node, rewriter.ctx)
  assert res is node


def test_too_many_args(rewriter: PivotRewriter, monkeypatch: pytest.MonkeyPatch) -> None:
  """Verifies that the method is not rewritten if it has multiple args.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      monkeypatch: Pytest monkeypatch fixture.
  """
  monkeypatch.setattr(rewriter.ctx, "resolve_type", MagicMock(return_value="Tensor"), raising=False)
  monkeypatch.setattr(rewriter.ctx, "lookup_api", MagicMock(return_value="shape"), raising=False)
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("size")),
    args=[cst.Arg(cst.Integer("0")), cst.Arg(cst.Integer("1"))],
  )
  res: cst.CSTNode | cst.Subscript | cst.Attribute = transform_method_to_property(node, rewriter.ctx)
  assert res is node


def test_data_ptr_mapping(rewriter: PivotRewriter) -> None:
  """Verify `Call` handles data ptr mapping as expected.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("data_ptr")))
  res: cst.CSTNode | cst.Subscript | cst.Attribute = transform_method_to_property(node, rewriter.ctx)
  assert isinstance(res, cst.Attribute)
  assert res.attr.value == "data"


def test_func_not_attribute(rewriter: PivotRewriter) -> None:
  """Verifies the behavior when node.func is not an Attribute.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  node: cst.Call = cst.Call(func=cst.Name("size"))
  res: cst.CSTNode | cst.Subscript | cst.Attribute = transform_method_to_property(node, rewriter.ctx)
  assert res is node


def test_unknown_method(rewriter: PivotRewriter) -> None:
  """Verifies the behavior when the method name is not recognized.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("unknown_method")))
  res: cst.CSTNode | cst.Subscript | cst.Attribute = transform_method_to_property(node, rewriter.ctx)
  assert res is node


def test_method_property_not_attribute() -> None:
  """Verify `Call` handles method property not attribute as expected."""
  node: cst.Call = cst.Call(func=cst.Name("size"))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: cst.CSTNode | cst.Subscript | cst.Attribute = transform_method_to_property(node, ctx)
  assert res is node


def test_method_property_unknown_method() -> None:
  """Verify `Call` handles method property unknown method as expected."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("unknown")))
  ctx: HookContext = HookContext(semantics=MagicMock(), config=MagicMock())
  res: cst.CSTNode | cst.Subscript | cst.Attribute = transform_method_to_property(node, ctx)
  assert res is node
