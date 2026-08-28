"""Test suite for the Method Property module."""

import pytest
import libcst as cst
from typing import Generator, Dict, Any, Optional, Tuple, Union
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.method_property import transform_method_to_property


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
def rewriter() -> Generator[PivotRewriter, None, None]:
  """Provides a mock rewriter for testing.

  Yields:
      PivotRewriter: A mock rewriter instance.
  """
  hooks._HOOKS["method_to_property"] = transform_method_to_property
  hooks._PLUGINS_LOADED = True
  mgr: MagicMock = MagicMock()
  size_def: Dict[str, Any] = {"variants": {"jax": {"api": "shape", "requires_plugin": "method_to_property"}}}
  data_ptr_def: Dict[str, Any] = {"variants": {"jax": {"api": "data", "requires_plugin": "method_to_property"}}}
  all_defs: Dict[str, Dict[str, Any]] = {"size": size_def, "data_ptr": data_ptr_def}

  def get_def_side_effect(name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Gets def side effect.

    Args:
        name (str): Definition name.

    Returns:
        Optional[Tuple[str, Dict[str, Any]]]: Definition or None.
    """
    if name == "size" or name.endswith(".size"):
      return ("size", size_def)
    return None

  mgr.get_definition.side_effect = get_def_side_effect
  mgr.get_known_apis.return_value = all_defs

  def resolve_variant_side_effect(aid: str, fw: str) -> Optional[Dict[str, Any]]:
    """Resolves variant side effect.

    Args:
        aid (str): Definition ID.
        fw (str): Target framework string.

    Returns:
        Optional[Dict[str, Any]]: Variant definition.
    """
    if aid in all_defs:
      return all_defs[aid]["variants"].get(fw)
    return None

  mgr.resolve_variant.side_effect = resolve_variant_side_effect
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}
  cfg: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework="jax")
  yield PivotRewriter(mgr, cfg)


def test_simple_size_conversion(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of simple size conversion.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  assert "x.shape" in rewrite_code(rewriter, "s = x.size()")


def test_indexed_size_conversion(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of indexed size conversion.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  assert "x.shape[0]" in rewrite_code(rewriter, "d = x.size(0)").replace(" ", "")


def test_ignore_other_methods(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of ignore other methods.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  assert "x.other()" in rewrite_code(rewriter, "x.other()")


def test_obj_type_not_tensor(rewriter: PivotRewriter) -> None:
  """Verifies that the method is not rewritten if the receiver is known to not be a tensor.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  rewriter.ctx.resolve_type = MagicMock(return_value="Module")
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("size")))
  res: Union[cst.CSTNode, cst.Subscript, cst.Attribute] = transform_method_to_property(node, rewriter.ctx)
  assert res is node


def test_missing_target_prop(rewriter: PivotRewriter) -> None:
  """Verifies that the method is not rewritten if lookup_api fails.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  rewriter.ctx.resolve_type = MagicMock(return_value="Tensor")
  rewriter.ctx.lookup_api = MagicMock(return_value=None)
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("size")))
  res: Union[cst.CSTNode, cst.Subscript, cst.Attribute] = transform_method_to_property(node, rewriter.ctx)
  assert res is node


def test_too_many_args(rewriter: PivotRewriter) -> None:
  """Verifies that the method is not rewritten if it has multiple args.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  rewriter.ctx.resolve_type = MagicMock(return_value="Tensor")
  rewriter.ctx.lookup_api = MagicMock(return_value="shape")
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("size")),
    args=[cst.Arg(cst.Integer("0")), cst.Arg(cst.Integer("1"))],
  )
  res: Union[cst.CSTNode, cst.Subscript, cst.Attribute] = transform_method_to_property(node, rewriter.ctx)
  assert res is node


def test_data_ptr_mapping(rewriter: PivotRewriter) -> None:
  """Verifies the behavior of data ptr mapping.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("data_ptr")))
  res: Union[cst.CSTNode, cst.Subscript, cst.Attribute] = transform_method_to_property(node, rewriter.ctx)
  assert isinstance(res, cst.Attribute)
  assert res.attr.value == "data"


def test_func_not_attribute(rewriter: PivotRewriter) -> None:
  """Verifies the behavior when node.func is not an Attribute.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  node: cst.Call = cst.Call(func=cst.Name("size"))
  res: Union[cst.CSTNode, cst.Subscript, cst.Attribute] = transform_method_to_property(node, rewriter.ctx)
  assert res is node


def test_unknown_method(rewriter: PivotRewriter) -> None:
  """Verifies the behavior when the method name is not recognized.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
  """
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("unknown_method")))
  res: Union[cst.CSTNode, cst.Subscript, cst.Attribute] = transform_method_to_property(node, rewriter.ctx)
  assert res is node
