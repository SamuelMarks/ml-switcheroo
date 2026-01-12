"""
Tests for State Flag Injection Plugin (Eval/Train mode).

Verifies:
1.  **State Capture**: `model.eval()` is stripped and state recorded in Context.
2.  **Flag Injection**: `model(x)` gets `training=False` injected if state detected.
3.  **Isolation**: Distinct rewriters do not share state.
4.  **Scope**: State tracks the specific object instance name.
"""

import pytest
import libcst as cst
from unittest.mock import MagicMock

# Correctly import the Test shim instead of the deleted core class
from tests.conftest import TestRewriter as PivotRewriter

from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.state_flag_injection import inject_training_flag_call, capture_eval_state


def rewrite(rewriter, code):
  """Executes the rewriter pipeline on the code."""
  # Use rewriter.convert() because TestRewriter wraps the pipeline, it is not a direct Visitor
  tree = cst.parse_module(code)
  new_tree = rewriter.convert(tree)
  return new_tree.code


@pytest.fixture
def rewriter_factory():
  hooks._HOOKS["inject_training_flag"] = inject_training_flag_call
  hooks._HOOKS["capture_eval_state"] = capture_eval_state
  hooks._PLUGINS_LOADED = True

  mgr = MagicMock()
  eval_def = {"variants": {"jax": {"requires_plugin": "capture_eval_state"}}}
  call_def = {"variants": {"jax": {"requires_plugin": "inject_training_flag"}}}

  def resolve(aid, fw):
    if aid == "eval":
      return eval_def["variants"]["jax"]
    if aid == "call":
      return call_def["variants"]["jax"]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_definition.side_effect = lambda n: (("eval", eval_def) if "eval" in n or "train" in n else ("call", call_def))
  # Ensure verification logic passes
  mgr.is_verified.return_value = True

  # Ensure attribute access passes
  mgr.get_framework_config.return_value = {}

  def create():
    return PivotRewriter(mgr, RuntimeConfig(source_framework="torch", target_framework="jax"))

  return create


def test_state_isolation(rewriter_factory):
  """Verify that state captured in one rewriter doesn't leak to another."""
  r1 = rewriter_factory()
  r2 = rewriter_factory()

  rewrite(r1, "m.eval()")
  res2 = rewrite(r2, "m(x)")
  assert "training=False" not in res2

  res1 = rewrite(r1, "m(x)")
  assert "training=False" in res1


def test_eval_stripping(rewriter_factory):
  """Verify .eval() call is replaced by None (No-op)."""
  r = rewriter_factory()
  res = rewrite(r, "m.eval()")
  assert "None" in res or "pass" in res or res.strip() == ""


def test_state_injection_eval(rewriter_factory):
  """Verify training=False injection after eval()."""
  r = rewriter_factory()
  res = rewrite(r, "m.eval(); m(x)")
  assert "training=False" in res


def test_state_injection_train(rewriter_factory):
  """Verify training=True injection after train()."""
  r = rewriter_factory()
  res = rewrite(r, "m.train(); m(x)")
  assert "training=True" in res


def test_scope_isolation(rewriter_factory):
  """Verify flags track specific object names (m1 vs m2)."""
  r = rewriter_factory()
  res = rewrite(r, "m1.eval(); m2.train(); m1(x); m2(x)")
  assert "m1(x, training=False)" in res
  assert "m2(x, training=True)" in res


def test_attribute_resolution(rewriter_factory):
  """Verify dot-separated names are tracked correctly."""
  r = rewriter_factory()
  res = rewrite(r, "self.l.eval(); self.l(x)")
  assert "training=False" in res
