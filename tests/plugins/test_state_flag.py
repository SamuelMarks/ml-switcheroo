"""Test suite for the State Flag module."""

import typing
from unittest.mock import MagicMock

import libcst as cst
import pytest
from pydantic import JsonValue

import ml_switcheroo.core.hooks as hooks
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.plugins.state_flag_injection import _get_func_name, capture_eval_state, inject_training_flag_call
from tests.conftest import TestRewriter as PivotRewriter


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
def rewriter_factory() -> typing.Callable[[], PivotRewriter]:
  """Provides a mock rewriter factory for testing.

  Returns:
      Callable[[], PivotRewriter]: Factory function.
  """
  hooks.register_hook("inject_training_flag")(inject_training_flag_call)
  hooks.register_hook("capture_eval_state")(capture_eval_state)
  hooks.set_plugins_loaded(True)
  mgr: MagicMock = MagicMock()
  eval_jax: dict[str, JsonValue] = {"requires_plugin": "capture_eval_state"}
  call_jax: dict[str, JsonValue] = {"requires_plugin": "inject_training_flag"}
  eval_def = {"variants": {"jax": eval_jax}}
  call_def = {"variants": {"jax": call_jax}}

  def resolve(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Resolves variant.

    Args:
        aid (str): Definition ID.
        fw (str): Target framework string.

    Returns:
        dict[str, JsonValue] | None: Variant definition.
    """
    if aid == "eval":
      _ret_v_1: dict[str, JsonValue] | None = eval_def["variants"]["jax"]
      return _ret_v_1
    if aid == "call":
      _ret_v_2: dict[str, JsonValue] | None = call_def["variants"]["jax"]
      return _ret_v_2
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


def test_state_isolation(rewriter_factory: typing.Callable[[], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `training=False` for state isolation.

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


def test_eval_stripping(rewriter_factory: typing.Callable[[], PivotRewriter]) -> None:
  """Verify `rewriter_factory` handles eval stripping as expected.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "m.eval()")
  assert "None" in res or "pass" in res or res.strip() == ""


def test_state_injection_eval(rewriter_factory: typing.Callable[[], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `training=False` for state injection eval.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "m.eval(); m(x)")
  assert "training=False" in res


def test_state_injection_train(rewriter_factory: typing.Callable[[], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `training=True` for state injection train.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "m.train(); m(x)")
  assert "training=True" in res


def test_scope_isolation(rewriter_factory: typing.Callable[[], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `m1(x, training=False)` for scope isolation.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "m1.eval(); m2.train(); m1(x); m2(x)")
  assert "m1(x, training=False)" in res
  assert "m2(x, training=True)" in res


def test_attribute_resolution(rewriter_factory: typing.Callable[[], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `training=False` for attribute resolution.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "self.l.eval(); self.l(x)")
  assert "training=False" in res


def test_missing_state_returns_node(rewriter_factory: typing.Callable[[], PivotRewriter]) -> None:
  """Verifies that missing state doesn't crash but returns the original node.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  # m1.eval() populates store for m1, but we call m2
  res: str = rewrite(r, "m1.eval(); m2(x)")
  assert "m2(x)" in res
  assert "training=" not in res


def test_kwarg_already_exists(rewriter_factory: typing.Callable[[], PivotRewriter]) -> None:
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
  res: cst.CSTNode | cst.Name = capture_eval_state(node, ctx)
  assert res is node


def test_unsupported_receiver_name() -> None:
  """Verify `Call` handles unsupported receiver name as expected."""
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
  res: cst.CSTNode | cst.Name = capture_eval_state(node, ctx)
  assert res is node


def test_train_with_args(rewriter_factory: typing.Callable[[], PivotRewriter]) -> None:
  """Verify `rewriter_factory` emits `training=mode_var` for train with args.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "m.train(mode_var); m(x)")
  assert "training=mode_var" in res


def test_get_func_name() -> None:
  """Gets function name."""
  assert _get_func_name(cst.Call(func=cst.SimpleString("'a'"))) is None


def test_inject_training_flag_call() -> None:
  """Injects training flag call."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.metadata = {}
  ctx.lookup_api.return_value = None
  node1: cst.Call = cst.Call(func=cst.Name("dropout"))
  assert inject_training_flag_call(node1, ctx) == node1
  ctx.lookup_api.return_value = "Dropout"
  node_pos: cst.Call = cst.Call(
    func=cst.Name("dropout"), args=[cst.Arg(value=cst.Name("x")), cst.Arg(value=cst.Name("training"))]
  )
  inject_training_flag_call(node_pos, ctx)


def test_capture_eval_state() -> None:
  """Verify `Call` handles capture eval state as expected."""
  ctx: MagicMock = MagicMock(spec=HookContext)
  ctx.metadata = {}
  node1: cst.Call = cst.Call(func=cst.Name("foo"))
  assert capture_eval_state(node1, ctx) == node1
  node_not_method: cst.Call = cst.Call(func=cst.Name("eval"))
  assert capture_eval_state(node_not_method, ctx) == node_not_method
  node_train_pos: cst.Call = cst.Call(
    func=cst.Attribute(value=cst.Name("model"), attr=cst.Name("train")), args=[cst.Arg(value=cst.Name("False"))]
  )
  capture_eval_state(node_train_pos, ctx)
