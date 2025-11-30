"""
Tests for State Flag Injection Plugin (Context-Aware).

Verifies that:
1. `model.eval()` calls are captured and updates HookContext metadata.
2. Subsequent `model(x)` calls receive `training=False` via metadata reading.
3. Multiple contexts maintain separate state stacks (Thread Safety).
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.state_flag_injection import (
  inject_training_flag_call,
  capture_eval_state,
)


@pytest.fixture
def rewriter_factory():
  """Factory to create rewriters with fresh contexts."""
  # 1. Register Hooks
  hooks._HOOKS["inject_training_flag"] = inject_training_flag_call
  hooks._HOOKS["capture_eval_state"] = capture_eval_state
  hooks._PLUGINS_LOADED = True

  # 2. Mock Semantics
  mgr = MagicMock()

  # Define behavior lookup
  def get_def(name):
    if "eval" in name or "train" in name:
      return (
        "eval_op",
        {
          "variants": {
            "jax": {"requires_plugin": "capture_eval_state"},
            "torch": {"api": name},
          }
        },
      )
    # We assume generic objects 'model' resolve to something that uses injection
    return (
      "call_op",
      {
        "variants": {
          "jax": {"requires_plugin": "inject_training_flag"},
          "torch": {"api": name},
        }
      },
    )

  mgr.get_definition.side_effect = get_def
  mgr.is_verified.return_value = True

  def create():
    cfg = RuntimeConfig(source_framework="torch", target_framework="jax")
    return PivotRewriter(semantics=mgr, config=cfg)

  return create


def rewrite_code_with_context_check(rewriter, code):
  """Executes rewrite."""
  tree = cst.parse_module(code)
  try:
    new_tree = tree.visit(rewriter)
    return new_tree.code
  except Exception as e:
    pytest.fail(f"Rewrite failed: {e}")


def test_state_isolation(rewriter_factory):
  """
  Verify that state doesn't leak between two different rewriter instances.
  """
  rewriter1 = rewriter_factory()
  rewriter2 = rewriter_factory()

  code1 = "model.eval()"
  code2 = "model(x)"

  # Rewriter 1 processes eval(), so it knows model is eval mode
  rewrite_code_with_context_check(rewriter1, code1)

  # Rewriter 2 processes model(x), but it didn't see eval(), so it shouldn't inject
  res2 = rewrite_code_with_context_check(rewriter2, code2)

  # Check isolation
  assert "training=False" not in res2

  # Verify Rewriter 1 would have injected it
  res1 = rewrite_code_with_context_check(rewriter1, code2)
  assert "training=False" in res1


def test_eval_stripping(rewriter_factory):
  """
  Input: model.eval()
  Output: None()  (effectively no-op)
  """
  rewriter = rewriter_factory()
  code = "model.eval()"
  result = rewrite_code_with_context_check(rewriter, code)
  assert "None()" in result
  assert "model.eval()" not in result


def test_state_injection_eval(rewriter_factory):
  """
  Scenario:
      model.eval()
      model(x)
  Expect:
      None()
      model(x, training=False)
  """
  rewriter = rewriter_factory()
  code = """
model.eval()
y = model(x)
"""
  result = rewrite_code_with_context_check(rewriter, code)

  assert "None()" in result
  assert "training=False" in result


def test_state_injection_train(rewriter_factory):
  """
  Scenario:
      model.train()
      model(x)
  """
  rewriter = rewriter_factory()
  code = """
model.train()
y = model(x)
"""
  result = rewrite_code_with_context_check(rewriter, code)
  assert "training=True" in result


def test_scope_isolation_multiple_objects(rewriter_factory):
  """
  Scenario:
      m1.eval()
      m2.train()
      m1(x)
      m2(x)
  Expect:
      m1 has training=False
      m2 has training=True
  """
  rewriter = rewriter_factory()
  code = """
m1.eval()
m2.train()
a = m1(x)
b = m2(x)
"""
  result = rewrite_code_with_context_check(rewriter, code)

  assert "m1(x, training=False)" in result
  assert "m2(x, training=True)" in result


def test_attribute_resolution(rewriter_factory):
  """
  Scenario: self.layer.eval() -> self.layer(x, training=False)
  """
  rewriter = rewriter_factory()
  code = """
self.layer.eval()
y = self.layer(x)
"""
  result = rewrite_code_with_context_check(rewriter, code)
  assert "training=False" in result
