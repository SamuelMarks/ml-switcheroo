"""Test suite for guards.py"""

import libcst as cst
import typing
from unittest.mock import MagicMock

from ml_switcheroo.core.rewriter.calls.guards import apply_strict_guards


class MockContext:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.hook_context = MagicMock()
    self.hook_context.metadata = {}


class MockRewriter:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.context = MockContext()


def parse_arg(code: str) -> cst.Arg:
  """Docstring."""
  module = cst.parse_module(code)
  # Extract arg from "foo(a=1)" -> `a=1`
  expr = typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]).value
  return typing.cast(cst.Call, expr).args[0]


def test_apply_strict_guards_no_guards() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  args: list[cst.Arg] = [parse_arg("foo(x=1)")]
  details: dict[str, typing.Any] = {}
  target_impl: dict[str, typing.Any] = {}

  result: list[cst.Arg] = apply_strict_guards(rewriter, args, details, target_impl)  # type: ignore
  assert result == args


def test_apply_strict_guards_match() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  args = [parse_arg("foo(x=1)")]

  details: dict[str, typing.Any] = {"std_args": [{"name": "x", "rank": 2}]}
  target_impl: dict[str, typing.Any] = {"args": {"x": "x"}}

  result: list[cst.Arg] = apply_strict_guards(rewriter, args, details, target_impl)  # type: ignore
  assert len(result) == 1

  # Value should be wrapped in _check_rank call
  assert isinstance(result[0].value, cst.Call)
  assert typing.cast(cst.Name, result[0].value.func).value == "_check_rank"
  assert typing.cast(cst.Integer, result[0].value.args[1].value).value == "2"

  # Should inject preamble
  rewriter.context.hook_context.inject_preamble.assert_called_once()
  assert rewriter.context.hook_context.metadata.get("strict_helper_injected")


def test_apply_strict_guards_match_different_name() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  args = [parse_arg("foo(input_tensor=1)")]

  details: dict[str, typing.Any] = {"std_args": [{"name": "x", "rank": 3}]}
  target_impl: dict[str, typing.Any] = {"args": {"x": "input_tensor"}}

  result: list[cst.Arg] = apply_strict_guards(rewriter, args, details, target_impl)  # type: ignore
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)
  assert typing.cast(cst.Name, result[0].value.func).value == "_check_rank"
  assert typing.cast(cst.Integer, result[0].value.args[1].value).value == "3"


def test_apply_strict_guards_no_match() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  args = [parse_arg("foo(y=1)")]

  details: dict[str, typing.Any] = {"std_args": [{"name": "x", "rank": 2}]}
  target_impl: dict[str, typing.Any] = {"args": {"x": "x"}}

  result: list[cst.Arg] = apply_strict_guards(rewriter, args, details, target_impl)  # type: ignore
  assert result == args


def test_apply_strict_guards_already_injected() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  rewriter.context.hook_context.metadata["strict_helper_injected"] = True

  args = [parse_arg("foo(x=1)")]
  details: dict[str, typing.Any] = {"std_args": [{"name": "x", "rank": 2}]}
  target_impl: dict[str, typing.Any] = {"args": {"x": "x"}}

  apply_strict_guards(rewriter, args, details, target_impl)  # type: ignore
  rewriter.context.hook_context.inject_preamble.assert_not_called()


def test_apply_strict_guards_no_context() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  del rewriter.context  # type: ignore
  args = [parse_arg("foo(x=1)")]
  details: dict[str, typing.Any] = {"std_args": [{"name": "x", "rank": 2}]}
  target_impl: dict[str, typing.Any] = {"args": {"x": "x"}}
  result: list[cst.Arg] = apply_strict_guards(rewriter, args, details, target_impl)  # type: ignore
  assert len(result) == 1


def test_apply_strict_guards_fallback_name() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  args = [parse_arg("foo(x=1)")]
  details: dict[str, typing.Any] = {"std_args": [{"name": "x", "rank": 2}]}
  target_impl: dict[str, typing.Any] = {}  # x is not in args mapping
  result: list[cst.Arg] = apply_strict_guards(rewriter, args, details, target_impl)  # type: ignore
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)
