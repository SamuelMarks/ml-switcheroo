"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock, patch
from ml_switcheroo.plugins.shape_packing import transform_shape_packing
from ml_switcheroo.core.hooks import HookContext


def test_transform_shape_packing_no_target_api():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "Reshape"
  ctx.lookup_api.return_value = None
  node = cst.parse_expression("x.view(1, 2, -1)")
  assert transform_shape_packing(node, ctx) is node

  ctx.current_op_id = "Custom"
  ctx.lookup_api.return_value = None
  assert transform_shape_packing(node, ctx) is node


@patch("ml_switcheroo.plugins.shape_packing.is_framework_module_node")
def test_transform_shape_packing_method_call(mock_is_framework):
  """Docstring."""
  mock_is_framework.return_value = False
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "Reshape"
  ctx.lookup_api.return_value = "jnp.reshape"

  # Multi args
  node = cst.parse_expression("x.view(1, 2, -1)")
  new_node = transform_shape_packing(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "jnp.reshape(x, (1, 2, -1))"

  # Single integer arg -> tuple (1,)
  node = cst.parse_expression("x.view(1)")
  new_node = transform_shape_packing(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "jnp.reshape(x, (1, ))"

  # Single list arg -> passed as is
  node = cst.parse_expression("x.view([1, 2])")
  new_node = transform_shape_packing(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "jnp.reshape(x, [1, 2])"

  # No args -> early return
  node = cst.parse_expression("x.view()")
  assert transform_shape_packing(node, ctx) is node


@patch("ml_switcheroo.plugins.shape_packing.is_framework_module_node")
def test_transform_shape_packing_function_call(mock_is_framework):
  """Docstring."""
  mock_is_framework.return_value = True
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "Reshape"
  ctx.lookup_api.return_value = "jnp.reshape"

  # Multi args
  node = cst.parse_expression("torch.reshape(x, 1, 2, -1)")
  new_node = transform_shape_packing(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "jnp.reshape(x, (1, 2, -1))"

  # No args -> early return
  node = cst.parse_expression("torch.reshape()")
  assert transform_shape_packing(node, ctx) is node

  # One arg (no shape) -> early return
  node = cst.parse_expression("torch.reshape(x)")
  assert transform_shape_packing(node, ctx) is node
