"""Test suite for the Shape Packing module."""

import pytest
import libcst as cst
from typing import Callable, Dict, Any, Union
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.shape_packing import transform_shape_packing
from ml_switcheroo.frameworks.base import register_framework


def rewrite_code(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      code (str): The code.

  Returns:
      str: The rewritten code string.
  """
  tree: cst.Module = cst.parse_module(code)
  return rewriter.convert(tree).code


@register_framework("custom_fw")
class CustomAdapter:
  """Test suite for the Custom Adapter component."""

  @property
  def harness_imports(self) -> list:
    """Helper to harness imports.

    Returns:
        list: List of imports.
    """
    return []

  def get_harness_init_code(self) -> str:
    """Gets harness initialization code.

    Returns:
        str: Empty string.
    """
    return ""

  def get_to_numpy_code(self) -> str:
    """Gets to NumPy code.

    Returns:
        str: Numpy code string.
    """
    return "return str(obj)"

  @property
  def declared_magic_args(self) -> list:
    """Helper to declared magic arguments.

    Returns:
        list: Empty list.
    """
    return []


@pytest.fixture
def rewriter_factory() -> Callable[[str], PivotRewriter]:
  """Provides a mock rewriter factory for testing.

  Returns:
      Callable[[str], PivotRewriter]: The factory function.
  """
  hooks._HOOKS["pack_shape_args"] = transform_shape_packing
  hooks._PLUGINS_LOADED = True
  mgr: MagicMock = MagicMock()
  def_map: Dict[str, Any] = {
    "variants": {
      "torch": {"api": "torch.view"},
      "jax": {"api": "jnp.reshape", "requires_plugin": "pack_shape_args"},
      "custom_fw": {"api": "custom.ops.reshape", "requires_plugin": "pack_shape_args"},
    }
  }
  mgr.get_known_apis.return_value = {"Reshape": def_map}

  def resolve(aid: str, fw: str) -> Any:
    """Resolves variant.

    Args:
        aid (str): Definition ID.
        fw (str): Target framework.

    Returns:
        Any: Variant definition.
    """
    if aid == "Reshape":
      return def_map["variants"].get(fw)
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_definition.side_effect = lambda n: ("Reshape", def_map) if "view" in n else None
  mgr.get_framework_config.return_value = {}

  def create(target: str) -> PivotRewriter:
    """Creates rewriter.

    Args:
        target (str): Target framework string.

    Returns:
        PivotRewriter: The rewriter.
    """
    cfg: RuntimeConfig = RuntimeConfig(source_framework="torch", target_framework=target)
    return PivotRewriter(mgr, cfg)

  return create


def test_packing_jax(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of packing JAX.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rw: PivotRewriter = rewriter_factory("jax")
  code: str = "y = x.view(1, 2)"
  res: str = rewrite_code(rw, code)
  assert "jnp.reshape(x" in res
  assert "(1, 2)" in res


def test_packing_custom_fw(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of packing custom framework.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rw: PivotRewriter = rewriter_factory("custom_fw")
  code: str = "y = x.view(1, 2)"
  res: str = rewrite_code(rw, code)
  assert "custom.ops.reshape(x" in res
  assert "(1, 2)" in res


def test_packing_missing_passthrough(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies the behavior of packing missing passthrough.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rw: PivotRewriter = rewriter_factory("numpy")
  code: str = "y = x.view(1, 2)"
  res: str = rewrite_code(rw, code)
  assert "x.view(1, 2)" in res


def test_packing_function_call(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies behavior when called as a function (e.g. torch.view) instead of a method.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rw: PivotRewriter = rewriter_factory("jax")
  code: str = "y = torch.view(x, 1, 2)"
  res: str = rewrite_code(rw, code)
  assert "jnp.reshape(x" in res
  assert "(1, 2)" in res


def test_packing_fallback_to_view() -> None:
  """Verifies behavior when Reshape is missing but View is present."""
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("view")), args=[cst.Arg(cst.Integer("1"))]
  )
  ctx: MagicMock = MagicMock()
  ctx.current_op_id = "Reshape"

  def lookup(op_id: str) -> Union[str, None]:
    """Lookup API.

    Args:
        op_id (str): API ID.

    Returns:
        Union[str, None]: API path.
    """
    if op_id == "Reshape":
      return None
    if op_id == "View":
      return "jnp.view_api"
    return None

  ctx.lookup_api.side_effect = lookup
  res: Union[cst.CSTNode, cst.Call] = transform_shape_packing(node, ctx)
  assert isinstance(res, cst.Call)
  assert isinstance(res.func, cst.Attribute)
  assert res.func.attr.value == "view_api"


def test_packing_missing_target_api() -> None:
  """Verifies behavior when target API is missing."""
  node: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("view")), args=[cst.Arg(cst.Integer("1"))]
  )
  ctx: MagicMock = MagicMock()
  ctx.current_op_id = "Reshape"
  ctx.lookup_api.return_value = None
  res: Union[cst.CSTNode, cst.Call] = transform_shape_packing(node, ctx)
  assert res is node


def test_packing_function_missing_args() -> None:
  """Verifies behavior when called as function without args."""
  node: cst.Call = cst.Call(func=cst.Name("torch_view"), args=[])
  ctx: MagicMock = MagicMock()
  ctx.current_op_id = "Reshape"
  ctx.lookup_api.return_value = "jnp.reshape"

  from unittest.mock import patch

  with patch("ml_switcheroo.plugins.shape_packing.is_framework_module_node", return_value=True):
    node = node.with_changes(func=cst.Attribute(value=cst.Name("torch"), attr=cst.Name("view")))
    res: Union[cst.CSTNode, cst.Call] = transform_shape_packing(node, ctx)
  assert res is node


def test_packing_method_missing_args() -> None:
  """Verifies behavior when called as method without shape args."""
  node: cst.Call = cst.Call(func=cst.Attribute(value=cst.Name("x"), attr=cst.Name("view")), args=[])
  ctx: MagicMock = MagicMock()
  ctx.current_op_id = "Reshape"
  ctx.lookup_api.return_value = "jnp.reshape"
  res: Union[cst.CSTNode, cst.Call] = transform_shape_packing(node, ctx)
  assert res is node


def test_packing_one_int_arg(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies behavior when shape is a single integer.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rw: PivotRewriter = rewriter_factory("jax")
  code: str = "y = x.view(1)"
  res: str = rewrite_code(rw, code)
  assert "jnp.reshape(x" in res
  assert "(1, )" in res


def test_packing_one_tuple_arg(rewriter_factory: Callable[[str], PivotRewriter]) -> None:
  """Verifies behavior when shape is already packed in a tuple/variable.

  Args:
      rewriter_factory (Callable[[str], PivotRewriter]): Factory instance.
  """
  rw: PivotRewriter = rewriter_factory("jax")
  code: str = "y = x.view(shape)"
  res: str = rewrite_code(rw, code)
  assert "jnp.reshape(x, shape)" in res
