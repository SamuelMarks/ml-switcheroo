"""Test module."""

from typing import Any, Dict, List

import libcst as cst

from ml_switcheroo.core.rewriter.calls.guards import STRICT_RANK_HELPER, apply_strict_guards


class MockHookContext:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.metadata: Dict[str, Any] = {}
    self.preamble_injected: List[str] = []

  def inject_preamble(self, text: str) -> None:
    """Docstring."""
    self.preamble_injected.append(text)


class MockContext:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.hook_context: MockHookContext = MockHookContext()


class MockRewriter:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.context: MockContext = MockContext()


def test_apply_strict_guards_no_guards() -> None:
  """Docstring."""
  rewriter: MockRewriter = MockRewriter()
  norm_args: List[cst.Arg] = [cst.Arg(value=cst.Name("x"))]
  details: Dict[str, Any] = {"std_args": ["a"]}
  target_impl: Dict[str, Any] = {"args": {}}

  result: List[cst.Arg] = apply_strict_guards(rewriter, norm_args, details, target_impl)
  assert result == norm_args


def test_apply_strict_guards_with_guards() -> None:
  """Docstring."""
  rewriter: MockRewriter = MockRewriter()
  norm_args: List[cst.Arg] = [
    cst.Arg(keyword=cst.Name("inputs"), value=cst.Name("x"), equal=cst.AssignEqual()),
    cst.Arg(value=cst.Name("y")),
  ]
  details: Dict[str, Any] = {"std_args": [{"name": "inputs", "rank": 2}, {"name": "other"}]}
  target_impl: Dict[str, Any] = {"args": {"inputs": "inputs"}}

  result: List[cst.Arg] = apply_strict_guards(rewriter, norm_args, details, target_impl)
  assert len(result) == 2

  # Check wrapper
  arg0: cst.Arg = result[0]
  assert isinstance(arg0.value, cst.Call)
  assert isinstance(arg0.value.func, cst.Name)
  assert arg0.value.func.value == "_check_rank"
  assert isinstance(arg0.value.args[0].value, cst.Name)
  assert arg0.value.args[0].value.value == "x"
  assert isinstance(arg0.value.args[1].value, cst.Integer)
  assert arg0.value.args[1].value.value == "2"

  assert result[1] == norm_args[1]

  # Check preamble
  assert rewriter.context.hook_context.metadata["strict_helper_injected"] is True
  assert rewriter.context.hook_context.preamble_injected == [STRICT_RANK_HELPER]


def test_apply_strict_guards_multiple_calls_preamble() -> None:
  """Docstring."""
  rewriter: MockRewriter = MockRewriter()
  rewriter.context.hook_context.metadata["strict_helper_injected"] = True

  norm_args: List[cst.Arg] = [cst.Arg(keyword=cst.Name("inputs"), value=cst.Name("x"), equal=cst.AssignEqual())]
  details: Dict[str, Any] = {"std_args": [{"name": "inputs", "rank": 3}]}
  target_impl: Dict[str, Any] = {"args": {"inputs": "inputs"}}

  result: List[cst.Arg] = apply_strict_guards(rewriter, norm_args, details, target_impl)
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)
  assert len(rewriter.context.hook_context.preamble_injected) == 0


def test_apply_strict_guards_target_impl_mapping() -> None:
  """Docstring."""
  rewriter: MockRewriter = MockRewriter()
  norm_args: List[cst.Arg] = [cst.Arg(keyword=cst.Name("target_arg_name"), value=cst.Name("x"), equal=cst.AssignEqual())]
  details: Dict[str, Any] = {"std_args": [{"name": "std_input", "rank": 4}]}
  target_impl: Dict[str, Any] = {"args": {"std_input": "target_arg_name"}}

  result: List[cst.Arg] = apply_strict_guards(rewriter, norm_args, details, target_impl)
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)
  assert getattr(result[0].value.args[1].value, "value", None) == "4"


def test_apply_strict_guards_no_context() -> None:
  """Docstring."""

  class RewriterNoContext:
    """Docstring."""

    pass

  rewriter: RewriterNoContext = RewriterNoContext()
  norm_args: List[cst.Arg] = [cst.Arg(keyword=cst.Name("inputs"), value=cst.Name("x"), equal=cst.AssignEqual())]
  details: Dict[str, Any] = {"std_args": [{"name": "inputs", "rank": 2}]}
  target_impl: Dict[str, Any] = {}

  result: List[cst.Arg] = apply_strict_guards(rewriter, norm_args, details, target_impl)
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)


def test_apply_strict_guards_guards_applied_false() -> None:
  """Docstring."""
  rewriter: MockRewriter = MockRewriter()
  norm_args: List[cst.Arg] = [cst.Arg(value=cst.Name("x"))]
  details: Dict[str, Any] = {"std_args": [{"name": "inputs", "rank": 2}]}
  target_impl: Dict[str, Any] = {}

  result: List[cst.Arg] = apply_strict_guards(rewriter, norm_args, details, target_impl)
  assert len(result) == 1
  assert result[0] == norm_args[0]


def test_apply_strict_guards_arg_key_in_guards_map() -> None:
  """Docstring."""
  rewriter: MockRewriter = MockRewriter()
  norm_args: List[cst.Arg] = [cst.Arg(keyword=cst.Name("inputs"), value=cst.Name("x"), equal=cst.AssignEqual())]
  details: Dict[str, Any] = {"std_args": [{"name": "inputs", "rank": 2}]}
  target_impl: Dict[str, Any] = {}

  result: List[cst.Arg] = apply_strict_guards(rewriter, norm_args, details, target_impl)
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)


def test_apply_strict_guards_arg_key_not_in_guards_map() -> None:
  """Docstring."""
  rewriter: MockRewriter = MockRewriter()
  norm_args: List[cst.Arg] = [cst.Arg(keyword=cst.Name("other"), value=cst.Name("x"), equal=cst.AssignEqual())]
  details: Dict[str, Any] = {"std_args": [{"name": "inputs", "rank": 2}]}
  target_impl: Dict[str, Any] = {}

  result: List[cst.Arg] = apply_strict_guards(rewriter, norm_args, details, target_impl)
  assert len(result) == 1
  assert result[0] == norm_args[0]
