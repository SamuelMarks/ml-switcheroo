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

  # Define "StepLR" with JAX mapping
  step_def = {
    "variants": {
      "torch": {"api": "torch.optim.lr_scheduler.StepLR"},
      "jax": {
        "api": "optax.custom_decay",  # Changed from default to verify dynamic lookup
        "requires_plugin": "scheduler_rewire",
        "args": {"step_size": "steps"},  # Verify custom arg mapping works
      },
      "keras": {
        "api": "keras.optimizers.schedules.ExponentialDecay",
        "requires_plugin": "scheduler_rewire",
        "args": {"step_size": "decay_steps", "initial_learning_rate": "initial_learning_rate"},
      },
    }
  }

  # Helper for definition lookup
  def get_def(name):
    if "StepLR" in name:
      return "StepLR", step_def
    return None

  mgr.get_definition.side_effect = get_def
  mgr.get_definition_by_id.side_effect = lambda aid: step_def if aid == "StepLR" else None

  # Helper for variant resolution
  def resolve(aid, fw):
    if aid == "StepLR" and fw in step_def["variants"]:
      return step_def["variants"][fw]
    return None

  mgr.resolve_variant.side_effect = resolve

  mgr.get_known_apis.return_value = {"StepLR": step_def}
  mgr.is_verified.return_value = True
  # Safe defaults
  mgr.get_framework_config.return_value = {}

  # Default to JAX
  cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
  return PivotRewriter(mgr, cfg)


def test_step_lr_rewire_jax(rewriter):
  """
  Input: StepLR(optimizer, step_size=30, gamma=0.1)
  Output: optax.custom_decay(init_value=1.0, steps=30, decay_rate=0.1, staircase=True)

  Verifies:
  1. API string 'optax.custom_decay' loaded from mock (not hardcoded).
  2. Arg rename 'step_size' -> 'steps' loaded from mock.
  """
  rewriter.ctx.current_op_id = "StepLR"

  code = "sched = StepLR(optimizer, step_size=30, gamma=0.1)"

  # Manual hook check
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value

  # Sync context
  rewriter.ctx.target_fw = "jax"

  res_node = transform_scheduler_init(call_node, rewriter.ctx)
  wrapper = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])])
  res = wrapper.code

  assert "optax.custom_decay" in res
  assert "init_value=1.0" in res
  assert "steps=30" in res  # Mapped via args dict
  assert "decay_rate=0.1" in res  # Default fallback
  assert "staircase=True" in res


def test_step_lr_retarget_keras(rewriter):
  """
  Scenario: Wire StepLR to Keras logic via Config change.

  Input: StepLR(optimizer, step_size=50)
  Output: keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1.0, decay_steps=50, staircase=True)

  Verifies argument remapping handled dynamically based on framework variant.
  """
  rewriter.ctx.current_op_id = "StepLR"
  rewriter.ctx.target_fw = "keras"

  code = "sched = StepLR(optimizer, step_size=50)"
  tree = cst.parse_module(code)
  call_node = tree.body[0].body[0].value

  res_node = transform_scheduler_init(call_node, rewriter.ctx)
  res = cst.Module(body=[cst.SimpleStatementLine([cst.Expr(res_node)])]).code

  assert "keras.optimizers.schedules.ExponentialDecay" in res
  assert "initial_learning_rate=1.0" in res
  assert "decay_steps=50" in res  # Mapped via args dict


def test_step_noop(rewriter):
  """
  Input: scheduler.step()
  Output: None
  """
  code = "scheduler.step()"
  # Manual hook
  ctx = MagicMock()
  ctx.target_fw = "jax"
  ctx.current_op_id = "step"

  node = cst.parse_expression(code)
  res = transform_scheduler_step(node, ctx)

  assert isinstance(res, cst.Name)
  assert res.value == "None"
