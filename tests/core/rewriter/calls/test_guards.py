"""Test suite for guards.py"""

import libcst as cst
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter.calls.guards import apply_strict_guards


class MockContext:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.hook_context = MagicMock()
    self.hook_context.metadata = {}


class MockRewriter:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.context = MockContext()


def parse_arg(code: str) -> cst.Arg:
  """Docstring."""
  module = cst.parse_module(code)
  # Extract arg from "foo(a=1)" -> `a=1`
  return module.body[0].body[0].value.args[0]


def test_apply_strict_guards_no_guards():
  """Docstring."""
  rewriter = MockRewriter()
  args = [parse_arg("foo(x=1)")]
  details = {}
  target_impl = {}

  result = apply_strict_guards(rewriter, args, details, target_impl)
  assert result == args


def test_apply_strict_guards_match():
  """Docstring."""
  rewriter = MockRewriter()
  args = [parse_arg("foo(x=1)")]

  details = {"std_args": [{"name": "x", "rank": 2}]}
  target_impl = {"args": {"x": "x"}}

  result = apply_strict_guards(rewriter, args, details, target_impl)
  assert len(result) == 1

  # Value should be wrapped in _check_rank call
  assert isinstance(result[0].value, cst.Call)
  assert result[0].value.func.value == "_check_rank"
  assert result[0].value.args[1].value.value == "2"

  # Should inject preamble
  rewriter.context.hook_context.inject_preamble.assert_called_once()
  assert rewriter.context.hook_context.metadata.get("strict_helper_injected")


def test_apply_strict_guards_match_different_name():
  """Docstring."""
  rewriter = MockRewriter()
  args = [parse_arg("foo(input_tensor=1)")]

  details = {"std_args": [{"name": "x", "rank": 3}]}
  target_impl = {"args": {"x": "input_tensor"}}

  result = apply_strict_guards(rewriter, args, details, target_impl)
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)
  assert result[0].value.func.value == "_check_rank"
  assert result[0].value.args[1].value.value == "3"


def test_apply_strict_guards_no_match():
  """Docstring."""
  rewriter = MockRewriter()
  args = [parse_arg("foo(y=1)")]

  details = {"std_args": [{"name": "x", "rank": 2}]}
  target_impl = {"args": {"x": "x"}}

  result = apply_strict_guards(rewriter, args, details, target_impl)
  assert result == args


def test_apply_strict_guards_already_injected():
  """Docstring."""
  rewriter = MockRewriter()
  rewriter.context.hook_context.metadata["strict_helper_injected"] = True

  args = [parse_arg("foo(x=1)")]
  details = {"std_args": [{"name": "x", "rank": 2}]}
  target_impl = {"args": {"x": "x"}}

  apply_strict_guards(rewriter, args, details, target_impl)
  rewriter.context.hook_context.inject_preamble.assert_not_called()


def test_apply_strict_guards_no_context():
  """Docstring."""
  rewriter = MockRewriter()
  del rewriter.context
  args = [parse_arg("foo(x=1)")]
  details = {"std_args": [{"name": "x", "rank": 2}]}
  target_impl = {"args": {"x": "x"}}
  result = apply_strict_guards(rewriter, args, details, target_impl)
  assert len(result) == 1


def test_apply_strict_guards_fallback_name():
  """Docstring."""
  rewriter = MockRewriter()
  args = [parse_arg("foo(x=1)")]
  details = {"std_args": [{"name": "x", "rank": 2}]}
  target_impl = {}  # x is not in args mapping
  result = apply_strict_guards(rewriter, args, details, target_impl)
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)
