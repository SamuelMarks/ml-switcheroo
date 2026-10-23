"""Test suite for rewriter calls guards."""

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.core.rewriter.calls.guards import STRICT_RANK_HELPER, apply_strict_guards


class MockHookContext:
  """Test helper class MockHookContext."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.metadata: dict[str, JsonValue] = {}
    self.preamble_injected: list[str] = []

  def inject_preamble(self, code_str: str) -> None:
    """Execute inject preamble helper logic.

    Args:
        code_str: Preamble string to inject.
    """
    self.preamble_injected.append(code_str)


class MockContext:
  """Test helper class MockContext."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.hook_context: MockHookContext = MockHookContext()


class MockRewriter:
  """Test helper class MockRewriter."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.context: MockContext = MockContext()


def test_apply_strict_guards_no_guards() -> None:
  """Verify apply strict guards no guards behavior."""
  rewriter: MockRewriter = MockRewriter()
  norm_args: list[cst.Arg] = [cst.Arg(value=cst.Name("x"))]
  details: dict[str, JsonValue] = {"std_args": ["a"]}
  target_impl: dict[str, JsonValue] = {"args": {}}

  result: list[cst.Arg] = apply_strict_guards(
    rewriter,
    norm_args,
    details,
    target_impl,
  )
  assert result == norm_args


def test_apply_strict_guards_with_guards() -> None:
  """Verify apply strict guards with guards behavior."""
  rewriter: MockRewriter = MockRewriter()
  norm_args: list[cst.Arg] = [
    cst.Arg(keyword=cst.Name("inputs"), value=cst.Name("x"), equal=cst.AssignEqual()),
    cst.Arg(value=cst.Name("y")),
  ]
  details = {"std_args": [{"name": "inputs", "rank": 2}, {"name": "other"}]}
  target_impl = {"args": {"inputs": "inputs"}}

  result: list[cst.Arg] = apply_strict_guards(
    rewriter,
    norm_args,
    details,
    target_impl,
  )
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
  """Verify apply strict guards multiple calls preamble behavior."""
  rewriter: MockRewriter = MockRewriter()
  rewriter.context.hook_context.metadata["strict_helper_injected"] = True

  norm_args: list[cst.Arg] = [cst.Arg(keyword=cst.Name("inputs"), value=cst.Name("x"), equal=cst.AssignEqual())]
  details = {"std_args": [{"name": "inputs", "rank": 3}]}
  target_impl = {"args": {"inputs": "inputs"}}

  result: list[cst.Arg] = apply_strict_guards(
    rewriter,
    norm_args,
    details,
    target_impl,
  )
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)
  assert len(rewriter.context.hook_context.preamble_injected) == 0


def test_apply_strict_guards_target_impl_mapping() -> None:
  """Verify apply strict guards target impl mapping behavior."""
  rewriter: MockRewriter = MockRewriter()
  norm_args: list[cst.Arg] = [cst.Arg(keyword=cst.Name("target_arg_name"), value=cst.Name("x"), equal=cst.AssignEqual())]
  details = {"std_args": [{"name": "std_input", "rank": 4}]}
  target_impl = {"args": {"std_input": "target_arg_name"}}

  result: list[cst.Arg] = apply_strict_guards(
    rewriter,
    norm_args,
    details,
    target_impl,
  )
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)
  assert getattr(result[0].value.args[1].value, "value", None) == "4"


def test_apply_strict_guards_no_context() -> None:
  """Verify apply strict guards no context behavior."""

  class RewriterNoContext(MockRewriter):
    """Test helper class RewriterNoContext."""

    def __init__(self) -> None:
      """Initialize without setting context attribute."""

  rewriter: RewriterNoContext = RewriterNoContext()
  norm_args: list[cst.Arg] = [cst.Arg(keyword=cst.Name("inputs"), value=cst.Name("x"), equal=cst.AssignEqual())]
  details: dict[str, JsonValue] = {"std_args": [{"name": "inputs", "rank": 2}]}
  target_impl: dict[str, JsonValue] = {}

  result: list[cst.Arg] = apply_strict_guards(
    rewriter,
    norm_args,
    details,
    target_impl,
  )
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)


def test_apply_strict_guards_guards_applied_false() -> None:
  """Verify apply strict guards guards applied false behavior."""
  rewriter: MockRewriter = MockRewriter()
  norm_args: list[cst.Arg] = [cst.Arg(value=cst.Name("x"))]
  details: dict[str, JsonValue] = {"std_args": [{"name": "inputs", "rank": 2}]}
  target_impl: dict[str, JsonValue] = {}

  result: list[cst.Arg] = apply_strict_guards(
    rewriter,
    norm_args,
    details,
    target_impl,
  )
  assert len(result) == 1
  assert result[0] == norm_args[0]


def test_apply_strict_guards_arg_key_in_guards_map() -> None:
  """Verify apply strict guards arg key in guards map behavior."""
  rewriter: MockRewriter = MockRewriter()
  norm_args: list[cst.Arg] = [cst.Arg(keyword=cst.Name("inputs"), value=cst.Name("x"), equal=cst.AssignEqual())]
  details: dict[str, JsonValue] = {"std_args": [{"name": "inputs", "rank": 2}]}
  target_impl: dict[str, JsonValue] = {}

  result: list[cst.Arg] = apply_strict_guards(
    rewriter,
    norm_args,
    details,
    target_impl,
  )
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)


def test_apply_strict_guards_arg_key_not_in_guards_map() -> None:
  """Verify apply strict guards arg key not in guards map behavior."""
  rewriter: MockRewriter = MockRewriter()
  norm_args: list[cst.Arg] = [cst.Arg(keyword=cst.Name("other"), value=cst.Name("x"), equal=cst.AssignEqual())]
  details: dict[str, JsonValue] = {"std_args": [{"name": "inputs", "rank": 2}]}
  target_impl: dict[str, JsonValue] = {}

  result: list[cst.Arg] = apply_strict_guards(
    rewriter,
    norm_args,
    details,
    target_impl,
  )
  assert len(result) == 1
  assert result[0] == norm_args[0]
