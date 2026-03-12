import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.shape_packing import _create_dotted_name, transform_shape_packing
from ml_switcheroo.core.hooks import HookContext


def test_create_dotted_name():
  res = _create_dotted_name("jax.numpy.reshape")
  assert isinstance(res, cst.Attribute)


def test_transform_shape_packing_misses():
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "Reshape"

  # line 56, 59: Reshape, View not found
  ctx.lookup_api.return_value = None
  node1 = cst.Call(func=cst.Name("reshape"))
  assert transform_shape_packing(node1, ctx) == node1

  # Not reshape/view (line 59)
  ctx.lookup_api.return_value = "foo"
  node2 = cst.Call(func=cst.Name("other"))
  assert transform_shape_packing(node2, ctx) == node2

  # function instead of method (line 68, 76-79)
  # This happens when node.func is Name
  ctx.lookup_api.side_effect = lambda x: "numpy.reshape" if x == "Reshape" else None

  # Missing args (line 76-77)
  node_func_no_args = cst.Call(func=cst.Name("reshape"), args=[])
  assert transform_shape_packing(node_func_no_args, ctx) == node_func_no_args

  # Function with 1 shape arg that is an integer (line 88-93)
  node_func_int = cst.Call(func=cst.Name("reshape"), args=[cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Integer("1"))])
  res_int = transform_shape_packing(node_func_int, ctx)

  # Function with 1 shape arg that is not integer (line 94-95)
  node_func_var = cst.Call(func=cst.Name("reshape"), args=[cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Tuple([]))])
  res_var = transform_shape_packing(node_func_var, ctx)

  # Function with NO shape arg (len == 0) -> line 97
  node_func_0 = cst.Call(func=cst.Name("reshape"), args=[cst.Arg(value=cst.Name("x"))])
  assert transform_shape_packing(node_func_0, ctx) == node_func_0
