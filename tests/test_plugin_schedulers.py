"""Test suite for plugin schedulers."""

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.schedulers import (
  _create_dotted_name,
  _get_target_arg_name,
  transform_scheduler_init,
  transform_scheduler_step,
)


def test_create_dotted_name() -> None:
  """Verify create dotted name behavior."""
  node: cst.BaseExpression = _create_dotted_name("optax.exponential_decay")
  assert isinstance(node, cst.Attribute)
  assert isinstance(node.value, cst.Name)
  assert node.value.value == "optax"
  assert node.attr.value == "exponential_decay"


def test_get_target_arg_name() -> None:
  """Verify get target arg name behavior."""
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
  """Verify transform scheduler init early return behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "StepLR"
  ctx.lookup_api.return_value = None

  node: cst.Call = cst.ensure_type(cst.parse_expression("StepLR(optimizer, step_size=30)"), cst.Call)
  assert transform_scheduler_init(node, ctx) is node

  ctx.lookup_api.return_value = "optax.piecewise_constant"
  ctx.current_op_id = "OtherLR"
  assert transform_scheduler_init(node, ctx) is node


def test_transform_scheduler_init_step_lr() -> None:
  """Verify transform scheduler init step lr behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "StepLR"
  ctx.lookup_api.return_value = "optax.piecewise_constant"
  ctx.current_variant.args = {"step_size": "transition_steps", "gamma": "decay_rate"}

  node: cst.Call = cst.ensure_type(cst.parse_expression("StepLR(optimizer, step_size=30, gamma=0.1)"), cst.Call)
  new_node: cst.BaseExpression = getattr(
    {
      "v": getattr({"v": transform_scheduler_init}, "__getitem__")("v")(
        getattr({"v": node}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "optax.piecewise_constant(init_value=1.0, transition_steps=30, decay_rate=0.1, staircase=True)"

  # Test positional
  node2: cst.Call = cst.ensure_type(cst.parse_expression("StepLR(optimizer, 30, 0.1)"), cst.Call)
  new_node2: cst.BaseExpression = getattr(
    {
      "v": getattr({"v": transform_scheduler_init}, "__getitem__")("v")(
        getattr({"v": node2}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  code2: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node2)])]).code
  assert code2.strip() == "optax.piecewise_constant(init_value=1.0, transition_steps=30, decay_rate=0.1, staircase=True)"

  # Test empty args
  node3: cst.Call = cst.ensure_type(cst.parse_expression("StepLR()"), cst.Call)
  new_node3: cst.BaseExpression = getattr(
    {
      "v": getattr({"v": transform_scheduler_init}, "__getitem__")("v")(
        getattr({"v": node3}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  code3: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node3)])]).code
  assert code3.strip() == "optax.piecewise_constant(init_value=1.0, staircase=True)"


def test_transform_scheduler_init_cosine_lr() -> None:
  """Verify transform scheduler init cosine lr behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.current_op_id = "CosineAnnealingLR"
  ctx.lookup_api.return_value = "optax.cosine_decay_schedule"
  ctx.current_variant.args = {"T_max": "decay_steps", "eta_min": "alpha"}

  node: cst.Call = cst.ensure_type(cst.parse_expression("CosineAnnealingLR(optimizer, T_max=50, eta_min=0.01)"), cst.Call)
  new_node: cst.BaseExpression = getattr(
    {
      "v": getattr({"v": transform_scheduler_init}, "__getitem__")("v")(
        getattr({"v": node}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  code: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node)])]).code
  assert code.strip() == "optax.cosine_decay_schedule(init_value=1.0, decay_steps=50, alpha=0.01)"

  # Test positional
  node2: cst.Call = cst.ensure_type(cst.parse_expression("CosineAnnealingLR(optimizer, 50, 0.01)"), cst.Call)
  new_node2: cst.BaseExpression = getattr(
    {
      "v": getattr({"v": transform_scheduler_init}, "__getitem__")("v")(
        getattr({"v": node2}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  code2: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node2)])]).code
  assert code2.strip() == "optax.cosine_decay_schedule(init_value=1.0, decay_steps=50, alpha=0.01)"

  # Test empty args
  node3: cst.Call = cst.ensure_type(cst.parse_expression("CosineAnnealingLR()"), cst.Call)
  new_node3: cst.BaseExpression = getattr(
    {
      "v": getattr({"v": transform_scheduler_init}, "__getitem__")("v")(
        getattr({"v": node3}, "__getitem__")("v"), getattr({"v": ctx}, "__getitem__")("v")
      )
    },
    "__getitem__",
  )("v")
  code3: str = cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=new_node3)])]).code
  assert "optax.cosine_decay_schedule(init_value=1.0" in code3.strip()


def test_transform_scheduler_step() -> None:
  """Verify transform scheduler step behavior."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  node: cst.Call = cst.ensure_type(cst.parse_expression("scheduler.step()"), cst.Call)
  new_node: cst.CSTNode = transform_scheduler_step(node, ctx)
  assert isinstance(new_node, cst.Name)
  assert new_node.value == "None"
