"""Docstring."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.plugins.schedulers import (
  _create_dotted_name,
  _get_target_arg_name,
  transform_scheduler_init,
  transform_scheduler_step,
)
from ml_switcheroo.core.hooks import HookContext


def test_create_dotted_name() -> None:
  """Docstring."""
  node: cst.BaseExpression = _create_dotted_name("optax.exponential_decay")
  assert isinstance(node, cst.Attribute)
  assert isinstance(node.value, cst.Name)
  assert node.value.value == "optax"
  assert node.attr.value == "exponential_decay"


def test_get_target_arg_name() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)

  # Case 1: no variant
  ctx.current_variant = None
  assert _get_target_arg_name(ctx, "std_name", "default") == "default"

  # Case 2: variant but no args
  ctx.current_variant = MagicMock()
  ctx.current_variant.args = None
  assert _get_target_arg_name(ctx, "std_name", "default") == "default"

  # Case 3: variant with args
  ctx.current_variant.args = {"std_name": "target_name"}
  assert _get_target_arg_name(ctx, "std_name", "default") == "target_name"


def test_transform_scheduler_init_early_return() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "StepLR"
  ctx.lookup_api.return_value = None

  node: cst.BaseExpression = cst.parse_expression("StepLR(optimizer, step_size=30)")
  assert transform_scheduler_init(node, ctx) is node

  ctx.lookup_api.return_value = "optax.piecewise_constant"
  ctx.current_op_id = "OtherLR"
  assert transform_scheduler_init(node, ctx) is node


def test_transform_scheduler_init_step_lr() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "StepLR"
  ctx.lookup_api.return_value = "optax.piecewise_constant"
  ctx.current_variant.args = {"step_size": "transition_steps", "gamma": "decay_rate"}

  node: cst.BaseExpression = cst.parse_expression("StepLR(optimizer, step_size=30, gamma=0.1)")
  new_node: cst.BaseExpression = transform_scheduler_init(node, ctx)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "optax.piecewise_constant(init_value=1.0, transition_steps=30, decay_rate=0.1, staircase=True)"

  # Test positional
  node2: cst.BaseExpression = cst.parse_expression("StepLR(optimizer, 30, 0.1)")
  new_node2: cst.BaseExpression = transform_scheduler_init(node2, ctx)
  code2: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node2)])]).code
  assert code2.strip() == "optax.piecewise_constant(init_value=1.0, transition_steps=30, decay_rate=0.1, staircase=True)"

  # Test empty args
  node3: cst.BaseExpression = cst.parse_expression("StepLR()")
  new_node3: cst.BaseExpression = transform_scheduler_init(node3, ctx)
  code3: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node3)])]).code
  assert code3.strip() == "optax.piecewise_constant(init_value=1.0, staircase=True)"


def test_transform_scheduler_init_cosine_lr() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "CosineAnnealingLR"
  ctx.lookup_api.return_value = "optax.cosine_decay_schedule"
  ctx.current_variant.args = {"T_max": "decay_steps", "eta_min": "alpha"}

  node: cst.BaseExpression = cst.parse_expression("CosineAnnealingLR(optimizer, T_max=50, eta_min=0.01)")
  new_node: cst.BaseExpression = transform_scheduler_init(node, ctx)
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "optax.cosine_decay_schedule(init_value=1.0, decay_steps=50, alpha=0.01)"

  # Test positional
  node2: cst.BaseExpression = cst.parse_expression("CosineAnnealingLR(optimizer, 50, 0.01)")
  new_node2: cst.BaseExpression = transform_scheduler_init(node2, ctx)
  code2: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node2)])]).code
  assert code2.strip() == "optax.cosine_decay_schedule(init_value=1.0, decay_steps=50, alpha=0.01)"

  # Test empty args
  node3: cst.BaseExpression = cst.parse_expression("CosineAnnealingLR()")
  new_node3: cst.BaseExpression = transform_scheduler_init(node3, ctx)
  code3: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node3)])]).code
  assert "optax.cosine_decay_schedule(init_value=1.0" in code3.strip()


def test_transform_scheduler_step() -> None:
  """Docstring."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  node: cst.BaseExpression = cst.parse_expression("scheduler.step()")
  new_node: cst.CSTNode = transform_scheduler_step(node, ctx)
  assert isinstance(new_node, cst.Name)
  assert new_node.value == "None"
