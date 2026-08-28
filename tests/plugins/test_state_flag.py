"""Test suite for the State Flag module."""

import pytest
import libcst as cst
from typing import Callable, Dict, Any, Union
from unittest.mock import MagicMock
from tests.conftest import TestRewriter as PivotRewriter
from ml_switcheroo.config import RuntimeConfig
import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.plugins.state_flag_injection import inject_training_flag_call, capture_eval_state


def rewrite(rewriter: PivotRewriter, code: str) -> str:
  """Rewrites code.

  Args:
      rewriter (PivotRewriter): The rewriter instance.
      code (str): The code to rewrite.

  Returns:
      str: The rewritten code string.
  """
  tree: cst.Module = cst.parse_module(code)
  new_tree: cst.Module = rewriter.convert(tree)
  return new_tree.code


@pytest.fixture
def rewriter_factory() -> Callable[[], PivotRewriter]:
  """Provides a mock rewriter factory for testing.

  Returns:
      Callable[[], PivotRewriter]: Factory function.
  """
  hooks._HOOKS["inject_training_flag"] = inject_training_flag_call
  hooks._HOOKS["capture_eval_state"] = capture_eval_state
  hooks._PLUGINS_LOADED = True
  mgr: MagicMock = MagicMock()
  eval_def: Dict[str, Any] = {"variants": {"jax": {"requires_plugin": "capture_eval_state"}}}
  call_def: Dict[str, Any] = {"variants": {"jax": {"requires_plugin": "inject_training_flag"}}}

  def resolve(aid: str, fw: str) -> Any:
    """Resolves variant.

    Args:
        aid (str): Definition ID.
        fw (str): Target framework string.

    Returns:
        Any: Variant definition.
    """
    if aid == "eval":
      return eval_def["variants"]["jax"]
    if aid == "call":
      return call_def["variants"]["jax"]
    return None

  mgr.resolve_variant.side_effect = resolve
  mgr.get_definition.side_effect = lambda n: ("eval", eval_def) if "eval" in n or "train" in n else ("call", call_def)
  mgr.is_verified.return_value = True
  mgr.get_framework_config.return_value = {}

  def create() -> PivotRewriter:
    """Creates rewriter.

    Returns:
        PivotRewriter: The rewriter.
    """
    return PivotRewriter(mgr, RuntimeConfig(source_framework="torch", target_framework="jax"))

  return create


def test_state_isolation(rewriter_factory: Callable[[], PivotRewriter]) -> None:
  """Verifies the behavior of state isolation.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r1: PivotRewriter = rewriter_factory()
  r2: PivotRewriter = rewriter_factory()
  rewrite(r1, "m.eval()")
  res2: str = rewrite(r2, "m(x)")
  assert "training=False" not in res2
  res1: str = rewrite(r1, "m(x)")
  assert "training=False" in res1


def test_eval_stripping(rewriter_factory: Callable[[], PivotRewriter]) -> None:
  """Verifies the behavior of eval stripping.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "m.eval()")
  assert "None" in res or "pass" in res or res.strip() == ""


def test_state_injection_eval(rewriter_factory: Callable[[], PivotRewriter]) -> None:
  """Verifies the behavior of state injection eval.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "m.eval(); m(x)")
  assert "training=False" in res


def test_state_injection_train(rewriter_factory: Callable[[], PivotRewriter]) -> None:
  """Verifies the behavior of state injection train.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "m.train(); m(x)")
  assert "training=True" in res


def test_scope_isolation(rewriter_factory: Callable[[], PivotRewriter]) -> None:
  """Verifies the behavior of scope isolation.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "m1.eval(); m2.train(); m1(x); m2(x)")
  assert "m1(x, training=False)" in res
  assert "m2(x, training=True)" in res


def test_attribute_resolution(rewriter_factory: Callable[[], PivotRewriter]) -> None:
  """Verifies the behavior of attribute resolution.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "self.l.eval(); self.l(x)")
  assert "training=False" in res


def test_missing_state_returns_node(rewriter_factory: Callable[[], PivotRewriter]) -> None:
  """Verifies that missing state doesn't crash but returns the original node.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  # m1.eval() populates store for m1, but we call m2
  res: str = rewrite(r, "m1.eval(); m2(x)")
  assert "m2(x)" in res
  assert "training=" not in res


def test_kwarg_already_exists(rewriter_factory: Callable[[], PivotRewriter]) -> None:
  """Verifies that the flag is not duplicated if already present.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "m.eval(); m(x, training=True)")
  assert "m(x, training=True)" in res
  # Ensure we don't have training=True, training=False
  assert res.count("training=") == 1


def test_unsupported_node_type_in_capture() -> None:
  """Verifies behavior when node.func is not an Attribute in capture_eval_state."""
  node: cst.Call = cst.Call(func=cst.Name("eval"), args=[])
  ctx: MagicMock = MagicMock()
  res: Union[cst.CSTNode, cst.Name] = capture_eval_state(node, ctx)
  assert res is node


def test_unsupported_receiver_name() -> None:
  """Verifies behavior when receiver name cannot be extracted."""
  # e.g., func_list[0].eval()
  node: cst.Call = cst.Call(
    func=cst.Attribute(
      value=cst.Subscript(
        value=cst.Name("func_list"), slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Integer("0")))]
      ),
      attr=cst.Name("eval"),
    ),
    args=[],
  )
  ctx: MagicMock = MagicMock()
  res: Union[cst.CSTNode, cst.Name] = capture_eval_state(node, ctx)
  assert res is node


def test_train_with_args(rewriter_factory: Callable[[], PivotRewriter]) -> None:
  """Verifies the behavior of train() with arguments.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "m.train(mode_var); m(x)")
  assert "training=mode_var" in res
