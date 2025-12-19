"""
Tests for Scheduler Rewiring Plugin.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.schedulers import transform_scheduler_init, transform_scheduler_step


def rewrite_code(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


@pytest.fixture
def rewriter():
  hooks._HOOKS["scheduler_rewire"] = transform_scheduler_init
  hooks._HOOKS["scheduler_step_noop"] = transform_scheduler_step
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()

  step_def = {
    "variants": {
      "torch": {"api": "torch.optim.lr_scheduler.StepLR"},
      "jax": {"api": "optax.expon", "requires_plugin": "scheduler_rewire"},
    }
  }

  mgr.get_definition.side_effect = lambda n: ("StepLR", step_def) if "StepLR" in n else None
  mgr.resolve_variant.side_effect = lambda aid, fw: step_def["variants"]["jax"]

  # We need current_op_id to be set for the logic to distinguish
  # Hooks logic relies on `ctx.current_op_id`
  # We will patch `rewrite_code` logic or the rewriter execution flow implicitly
  # via the side_effects/mocks in a real test environment, but here we just
  # manually invoke hook logic for unit testing specific functions if PivotRewriter integration is complex.

  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_step_lr_rewire(rewriter):
  """
  Input: StepLR(optimizer, step_size=30, gamma=0.1)
  Output: optax.exponential_decay(init_value=1.0, transition_steps=30, decay_rate=0.1, staircase=True)
  """
  # Mock OP ID context
  rewriter.ctx.current_op_id = "StepLR"

  code = "sched = StepLR(optimizer, step_size=30, gamma=0.1)"
  # Note: rewriter visit will call hook. We depend on mock resolving "StepLR" correctly from CST
  # PivotRewriter usually sets op_id if it finds a match.
  # To test logic specifically without setting up full semantic discovery for the node:

  # Manual hook invocation
  tree = cst.parse_module(code)
  # sched = StepLR(...) -> SimpleStatementLine -> Assign -> Call
  call_node = tree.body[0].body[0].value

  ctx = MagicMock()
  ctx.target_fw = "jax"
  ctx.current_op_id = "StepLR"

  res_node = transform_scheduler_init(call_node, ctx)
  wrapper = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])])
  res = wrapper.code

  assert "optax.exponential_decay" in res
  assert "init_value=1.0" in res
  assert "transition_steps=30" in res
  assert "decay_rate=0.1" in res
  assert "optimizer" not in res
  assert "staircase=True" in res


def test_cosine_lr_rewire(rewriter):
  code = "sched = CosineAnnealingLR(optimizer, T_max=50)"

  ctx = MagicMock()
  ctx.target_fw = "jax"
  ctx.current_op_id = "CosineAnnealingLR"

  call_node = cst.parse_expression("CosineAnnealingLR(optimizer, T_max=50)")
  res_node = transform_scheduler_init(call_node, ctx)

  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code

  assert "optax.cosine_decay_schedule" in res
  assert "decay_steps=50" in res
  assert "optimizer" not in res


def test_step_noop(rewriter):
  """
  Input: scheduler.step()
  Output: None
  """
  code = "scheduler.step()"
  # Manual hook
  ctx = MagicMock()
  ctx.target_fw = "jax"

  node = cst.parse_expression(code)
  res = transform_scheduler_step(node, ctx)

  assert isinstance(res, cst.Name)
  assert res.value == "None"
