"""Test suite for the Shape Packing Missing module."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.shape_packing import _create_dotted_name, transform_shape_packing
from ml_switcheroo.core.hooks import HookContext


def test_create_dotted_name():
  """Creates dotted name."""
  res = _create_dotted_name("jax.numpy.reshape")
  assert isinstance(res, cst.Attribute)


def test_transform_shape_packing_misses():
  """Transforms shape packing misses."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "Reshape"
  ctx.lookup_api.return_value = None
  node1 = cst.Call(func=cst.Name("reshape"))
  assert transform_shape_packing(node1, ctx) == node1
  ctx.lookup_api.return_value = "foo"
  node2 = cst.Call(func=cst.Name("other"))
  assert transform_shape_packing(node2, ctx) == node2
  ctx.lookup_api.side_effect = lambda x: "numpy.reshape" if x == "Reshape" else None
  node_func_no_args = cst.Call(func=cst.Name("reshape"), args=[])
  assert transform_shape_packing(node_func_no_args, ctx) == node_func_no_args
  node_func_int = cst.Call(func=cst.Name("reshape"), args=[cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Integer("1"))])
  transform_shape_packing(node_func_int, ctx)
  node_func_var = cst.Call(func=cst.Name("reshape"), args=[cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Tuple([]))])
  transform_shape_packing(node_func_var, ctx)
  node_func_0 = cst.Call(func=cst.Name("reshape"), args=[cst.Arg(value=cst.Name("x"))])
  assert transform_shape_packing(node_func_0, ctx) == node_func_0
