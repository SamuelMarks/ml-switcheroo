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


def test_create_dotted_name():
  """Docstring."""
  node = _create_dotted_name("optax.exponential_decay")
  assert isinstance(node, cst.Attribute)
  assert isinstance(node.value, cst.Name)
  assert node.value.value == "optax"
  assert node.attr.value == "exponential_decay"


def test_get_target_arg_name():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)

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


def test_transform_scheduler_init_early_return():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "StepLR"
  ctx.lookup_api.return_value = None

  node = cst.parse_expression("StepLR(optimizer, step_size=30)")
  assert transform_scheduler_init(node, ctx) is node

  ctx.lookup_api.return_value = "optax.piecewise_constant"
  ctx.current_op_id = "OtherLR"
  assert transform_scheduler_init(node, ctx) is node


def test_transform_scheduler_init_step_lr():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "StepLR"
  ctx.lookup_api.return_value = "optax.piecewise_constant"
  ctx.current_variant.args = {"step_size": "transition_steps", "gamma": "decay_rate"}

  node = cst.parse_expression("StepLR(optimizer, step_size=30, gamma=0.1)")
  new_node = transform_scheduler_init(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "optax.piecewise_constant(init_value=1.0, transition_steps=30, decay_rate=0.1, staircase=True)"

  # Test positional
  node = cst.parse_expression("StepLR(optimizer, 30, 0.1)")
  new_node = transform_scheduler_init(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "optax.piecewise_constant(init_value=1.0, transition_steps=30, decay_rate=0.1, staircase=True)"

  # Test empty args
  node = cst.parse_expression("StepLR()")
  new_node = transform_scheduler_init(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "optax.piecewise_constant(init_value=1.0, staircase=True)"


def test_transform_scheduler_init_cosine_lr():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  ctx.current_op_id = "CosineAnnealingLR"
  ctx.lookup_api.return_value = "optax.cosine_decay_schedule"
  ctx.current_variant.args = {"T_max": "decay_steps", "eta_min": "alpha"}

  node = cst.parse_expression("CosineAnnealingLR(optimizer, T_max=50, eta_min=0.01)")
  new_node = transform_scheduler_init(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "optax.cosine_decay_schedule(init_value=1.0, decay_steps=50, alpha=0.01)"

  # Test positional
  node = cst.parse_expression("CosineAnnealingLR(optimizer, 50, 0.01)")
  new_node = transform_scheduler_init(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "optax.cosine_decay_schedule(init_value=1.0, decay_steps=50, alpha=0.01)"

  # Test empty args
  node = cst.parse_expression("CosineAnnealingLR()")
  new_node = transform_scheduler_init(node, ctx)
  code = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert "optax.cosine_decay_schedule(init_value=1.0" in code.strip()


def test_transform_scheduler_step():
  """Docstring."""
  ctx = MagicMock(spec=HookContext)
  node = cst.parse_expression("scheduler.step()")
  new_node = transform_scheduler_step(node, ctx)
  assert isinstance(new_node, cst.Name)
  assert new_node.value == "None"
