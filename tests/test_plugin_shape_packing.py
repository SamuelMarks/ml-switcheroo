"""Docstring."""

from unittest.mock import MagicMock, patch

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.shape_packing import transform_shape_packing


def test_transform_shape_packing_no_target_api() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "Reshape"
  ctx.lookup_api.return_value = None
  node: cst.BaseExpression = cst.parse_expression("x.view(1, 2, -1)")
  assert transform_shape_packing(node, ctx) is node

  ctx.current_op_id = "Custom"
  ctx.lookup_api.return_value = None
  assert transform_shape_packing(node, ctx) is node


@patch("ml_switcheroo.plugins.shape_packing.is_framework_module_node")
def test_transform_shape_packing_method_call(mock_is_framework: MagicMock) -> None:
  """Docstring."""
  mock_is_framework.return_value = False
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "Reshape"
  ctx.lookup_api.return_value = "jnp.reshape"

  # Multi args
  node: cst.BaseExpression = cst.parse_expression("x.view(1, 2, -1)")
  new_node: cst.BaseExpression = transform_shape_packing(node, ctx)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "jnp.reshape(x, (1, 2, -1))"

  # Single integer arg -> tuple (1,)
  node2: cst.BaseExpression = cst.parse_expression("x.view(1)")
  new_node2: cst.BaseExpression = transform_shape_packing(node2, ctx)
  code2: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node2)])]).code
  assert code2.strip() == "jnp.reshape(x, (1, ))"

  # Single list arg -> passed as is
  node3: cst.BaseExpression = cst.parse_expression("x.view([1, 2])")
  new_node3: cst.BaseExpression = transform_shape_packing(node3, ctx)
  code3: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node3)])]).code
  assert code3.strip() == "jnp.reshape(x, [1, 2])"

  # No args -> early return
  node4: cst.BaseExpression = cst.parse_expression("x.view()")
  assert transform_shape_packing(node4, ctx) is node4


@patch("ml_switcheroo.plugins.shape_packing.is_framework_module_node")
def test_transform_shape_packing_function_call(mock_is_framework: MagicMock) -> None:
  """Docstring."""
  mock_is_framework.return_value = True
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "Reshape"
  ctx.lookup_api.return_value = "jnp.reshape"

  # Multi args
  node: cst.BaseExpression = cst.parse_expression("torch.reshape(x, 1, 2, -1)")
  new_node: cst.BaseExpression = transform_shape_packing(node, ctx)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "jnp.reshape(x, (1, 2, -1))"

  # No args -> early return
  node2: cst.BaseExpression = cst.parse_expression("torch.reshape()")
  assert transform_shape_packing(node2, ctx) is node2

  # One arg (no shape) -> early return
  node3: cst.BaseExpression = cst.parse_expression("torch.reshape(x)")
  assert transform_shape_packing(node3, ctx) is node3
