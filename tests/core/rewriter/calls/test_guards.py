"""Test suite for guards.py"""

from ml_switcheroo.enums import RuntimeTensorOrValue

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.rewriter.calls.guards import apply_strict_guards


class MockContext:
  """Test helper class MockContext."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.hook_context = MagicMock()
    self.hook_context.metadata = {}


class MockRewriter:
  """Test helper class MockRewriter."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.context = MockContext()


def parse_arg(code: str) -> cst.Arg:
  """Execute parse arg helper logic.

  Args:
      code: Python or DSL source code string to parse or transpile.

  Returns:
      Resulting `cst.Arg` from parse arg.
  """
  module = cst.parse_module(code)
  # Extract arg from "foo(a=1)" -> `a=1`
  expr = getattr(getattr(module.body[0], "body")[0], "value")
  return cst.ensure_type(expr, cst.Call).args[0]


def test_apply_strict_guards_no_guards() -> None:
  """Verify apply strict guards no guards behavior."""
  rewriter = MockRewriter()
  args: list[cst.Arg] = [parse_arg("foo(x=1)")]
  details: dict[str, RuntimeTensorOrValue] = {}
  target_impl: dict[str, RuntimeTensorOrValue] = {}

  result: list[cst.Arg] = apply_strict_guards(rewriter, args, details, target_impl)
  assert result == args


def test_apply_strict_guards_match() -> None:
  """Verify apply strict guards match behavior."""
  rewriter = MockRewriter()
  args = [parse_arg("foo(x=1)")]

  details = {"std_args": [{"name": "x", "rank": 2}]}
  target_impl = {"args": {"x": "x"}}

  result: list[cst.Arg] = apply_strict_guards(rewriter, args, details, target_impl)
  assert len(result) == 1

  # Value should be wrapped in _check_rank call
  assert isinstance(result[0].value, cst.Call)
  assert cst.ensure_type(result[0].value.func, cst.Name).value == "_check_rank"
  assert cst.ensure_type(result[0].value.args[1].value, cst.Integer).value == "2"

  # Should inject preamble
  rewriter.context.hook_context.inject_preamble.assert_called_once()
  assert rewriter.context.hook_context.metadata.get("strict_helper_injected")


def test_apply_strict_guards_match_different_name() -> None:
  """Verify apply strict guards match different name behavior."""
  rewriter = MockRewriter()
  args = [parse_arg("foo(input_tensor=1)")]

  details = {"std_args": [{"name": "x", "rank": 3}]}
  target_impl = {"args": {"x": "input_tensor"}}

  result: list[cst.Arg] = apply_strict_guards(rewriter, args, details, target_impl)
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)
  assert cst.ensure_type(result[0].value.func, cst.Name).value == "_check_rank"
  assert cst.ensure_type(result[0].value.args[1].value, cst.Integer).value == "3"


def test_apply_strict_guards_no_match() -> None:
  """Verify apply strict guards no match behavior."""
  rewriter = MockRewriter()
  args = [parse_arg("foo(y=1)")]

  details = {"std_args": [{"name": "x", "rank": 2}]}
  target_impl = {"args": {"x": "x"}}

  result: list[cst.Arg] = apply_strict_guards(rewriter, args, details, target_impl)
  assert result == args


def test_apply_strict_guards_already_injected() -> None:
  """Verify apply strict guards already injected behavior."""
  rewriter = MockRewriter()
  rewriter.context.hook_context.metadata["strict_helper_injected"] = True

  args = [parse_arg("foo(x=1)")]
  details = {"std_args": [{"name": "x", "rank": 2}]}
  target_impl = {"args": {"x": "x"}}

  apply_strict_guards(rewriter, args, details, target_impl)
  rewriter.context.hook_context.inject_preamble.assert_not_called()


def test_apply_strict_guards_no_context() -> None:
  """Verify apply strict guards no context behavior."""
  rewriter = MockRewriter()
  del rewriter.context
  args = [parse_arg("foo(x=1)")]
  details = {"std_args": [{"name": "x", "rank": 2}]}
  target_impl = {"args": {"x": "x"}}
  result: list[cst.Arg] = apply_strict_guards(rewriter, args, details, target_impl)
  assert len(result) == 1


def test_apply_strict_guards_fallback_name() -> None:
  """Verify apply strict guards fallback name behavior."""
  rewriter = MockRewriter()
  args = [parse_arg("foo(x=1)")]
  details = {"std_args": [{"name": "x", "rank": 2}]}
  target_impl: dict[str, RuntimeTensorOrValue] = {}  # x is not in args mapping
  result: list[cst.Arg] = apply_strict_guards(rewriter, args, details, target_impl)
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)
