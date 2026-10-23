"""Test suite for the State Flag module."""

import typing
from pydantic import JsonValue
from unittest.mock import MagicMock

import libcst as cst
import pytest

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
  getattr(getattr({"v": getattr({"v": hooks}, "__getitem__")("v")._HOOKS}, "__getitem__")("v"), "__setitem__")(
    "inject_training_flag", getattr({"v": inject_training_flag_call}, "__getitem__")("v")
  )
  getattr(getattr({"v": getattr({"v": hooks}, "__getitem__")("v")._HOOKS}, "__getitem__")("v"), "__setitem__")(
    "capture_eval_state", getattr({"v": capture_eval_state}, "__getitem__")("v")
  )
  hooks._PLUGINS_LOADED = True
  mgr: MagicMock = MagicMock()
  eval_def = {"variants": {"jax": {"requires_plugin": "capture_eval_state"}}}
  call_def = {"variants": {"jax": {"requires_plugin": "inject_training_flag"}}}

  def resolve(aid: str, fw: str) -> dict[str, JsonValue] | None:
    """Resolves variant.

    Args:
        aid (str): Definition ID.
        fw (str): Target framework string.

    Returns:
        Any: Variant definition.
    """
    if aid == "eval":
      _ret_v_1: dict[str, JsonValue] | None = getattr(
        {"v": getattr({"v": getattr({"v": eval_def}, "__getitem__")("v")["variants"]}, "__getitem__")("v")["jax"]},
        "__getitem__",
      )("v")
      return _ret_v_1
    if aid == "call":
      _ret_v_2: dict[str, JsonValue] | None = getattr(
        {"v": getattr({"v": getattr({"v": call_def}, "__getitem__")("v")["variants"]}, "__getitem__")("v")["jax"]},
        "__getitem__",
      )("v")
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


def test_eval_stripping(rewriter_factory: typing.Callable[[], PivotRewriter]) -> None:
  """Verifies the behavior of eval stripping.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "m.eval()")
  assert "None" in res or "pass" in res or res.strip() == ""


def test_state_injection_eval(rewriter_factory: typing.Callable[[], PivotRewriter]) -> None:
  """Verifies the behavior of state injection eval.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "m.eval(); m(x)")
  assert "training=False" in res


def test_state_injection_train(rewriter_factory: typing.Callable[[], PivotRewriter]) -> None:
  """Verifies the behavior of state injection train.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "m.train(); m(x)")
  assert "training=True" in res


def test_scope_isolation(rewriter_factory: typing.Callable[[], PivotRewriter]) -> None:
  """Verifies the behavior of scope isolation.

  Args:
      rewriter_factory (Callable[[], PivotRewriter]): Factory instance.
  """
  r: PivotRewriter = rewriter_factory()
  res: str = rewrite(r, "m1.eval(); m2.train(); m1(x); m2(x)")
  assert "m1(x, training=False)" in res
  assert "m2(x, training=True)" in res


def test_attribute_resolution(rewriter_factory: typing.Callable[[], PivotRewriter]) -> None:
  """Verifies the behavior of attribute resolution.

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
  """Verify unsupported receiver name behavior."""
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
  """Verifies the behavior of train() with arguments.

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
  """Verifies the behavior of capture eval state."""
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
