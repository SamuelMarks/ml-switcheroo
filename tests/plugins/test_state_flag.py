import pytest
import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter import PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.state_flag_injection import inject_training_flag_call, capture_eval_state


def rewrite(rewriter, code):
  return cst.parse_module(code).visit(rewriter).code


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
  mgr.get_definition.side_effect = lambda n: ("eval", eval_def) if "eval" in n or "train" in n else ("call", call_def)

  def create():
    return PivotRewriter(mgr, RuntimeConfig(source_framework="torch", target_framework="jax"))

  return create


def test_state_isolation(rewriter_factory):
  r1 = rewriter_factory()
  r2 = rewriter_factory()

  rewrite(r1, "m.eval()")
  res2 = rewrite(r2, "m(x)")
  assert "training=False" not in res2

  res1 = rewrite(r1, "m(x)")
  assert "training=False" in res1


def test_eval_stripping(rewriter_factory):
  r = rewriter_factory()
  res = rewrite(r, "m.eval()")
  assert "None()" in res


def test_state_injection_eval(rewriter_factory):
  r = rewriter_factory()
  res = rewrite(r, "m.eval(); m(x)")
  assert "training=False" in res


def test_state_injection_train(rewriter_factory):
  r = rewriter_factory()
  res = rewrite(r, "m.train(); m(x)")
  assert "training=True" in res


def test_scope_isolation(rewriter_factory):
  r = rewriter_factory()
  res = rewrite(r, "m1.eval(); m2.train(); m1(x); m2(x)")
  assert "m1(x, training=False)" in res
  assert "m2(x, training=True)" in res


def test_attribute_resolution(rewriter_factory):
  r = rewriter_factory()
  res = rewrite(r, "self.l.eval(); self.l(x)")
  assert "training=False" in res
