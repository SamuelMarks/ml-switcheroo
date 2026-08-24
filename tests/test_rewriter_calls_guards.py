"""Test module."""

import libcst as cst
from ml_switcheroo.core.rewriter.calls.guards import apply_strict_guards, STRICT_RANK_HELPER


class MockHookContext:
  """Test element."""

  def __init__(self):
    """Test element."""
    self.metadata = {}
    self.preamble_injected = []

  def inject_preamble(self, text):
    """Test element."""
    self.preamble_injected.append(text)


class MockContext:
  """Test element."""

  def __init__(self):
    """Test element."""
    self.hook_context = MockHookContext()


class MockRewriter:
  """Test element."""

  def __init__(self):
    """Test element."""
    self.context = MockContext()


def test_apply_strict_guards_no_guards():
  """Test element."""
  rewriter = MockRewriter()
  norm_args = [cst.Arg(value=cst.Name("x"))]
  details = {"std_args": ["a"]}
  target_impl = {"args": {}}

  result = apply_strict_guards(rewriter, norm_args, details, target_impl)
  assert result == norm_args


def test_apply_strict_guards_with_guards():
  """Test element."""
  rewriter = MockRewriter()
  norm_args = [
    cst.Arg(keyword=cst.Name("inputs"), value=cst.Name("x"), equal=cst.AssignEqual()),
    cst.Arg(value=cst.Name("y")),
  ]
  details = {"std_args": [{"name": "inputs", "rank": 2}, {"name": "other"}]}
  target_impl = {"args": {"inputs": "inputs"}}

  result = apply_strict_guards(rewriter, norm_args, details, target_impl)
  assert len(result) == 2

  # Check wrapper
  arg0 = result[0]
  assert isinstance(arg0.value, cst.Call)
  assert arg0.value.func.value == "_check_rank"
  assert arg0.value.args[0].value.value == "x"
  assert arg0.value.args[1].value.value == "2"

  assert result[1] == norm_args[1]

  # Check preamble
  assert rewriter.context.hook_context.metadata["strict_helper_injected"] is True
  assert rewriter.context.hook_context.preamble_injected == [STRICT_RANK_HELPER]


def test_apply_strict_guards_multiple_calls_preamble():
  """Test element."""
  rewriter = MockRewriter()
  rewriter.context.hook_context.metadata["strict_helper_injected"] = True

  norm_args = [cst.Arg(keyword=cst.Name("inputs"), value=cst.Name("x"), equal=cst.AssignEqual())]
  details = {"std_args": [{"name": "inputs", "rank": 3}]}
  target_impl = {"args": {"inputs": "inputs"}}

  result = apply_strict_guards(rewriter, norm_args, details, target_impl)
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)
  assert len(rewriter.context.hook_context.preamble_injected) == 0


def test_apply_strict_guards_target_impl_mapping():
  """Test element."""
  rewriter = MockRewriter()
  norm_args = [cst.Arg(keyword=cst.Name("target_arg_name"), value=cst.Name("x"), equal=cst.AssignEqual())]
  details = {"std_args": [{"name": "std_input", "rank": 4}]}
  target_impl = {"args": {"std_input": "target_arg_name"}}

  result = apply_strict_guards(rewriter, norm_args, details, target_impl)
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)
  assert result[0].value.args[1].value.value == "4"


def test_apply_strict_guards_no_context():
  """Test element."""

  class RewriterNoContext:
    pass

  rewriter = RewriterNoContext()
  norm_args = [cst.Arg(keyword=cst.Name("inputs"), value=cst.Name("x"), equal=cst.AssignEqual())]
  details = {"std_args": [{"name": "inputs", "rank": 2}]}
  target_impl = {}

  result = apply_strict_guards(rewriter, norm_args, details, target_impl)
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)


def test_apply_strict_guards_guards_applied_false():
  """Test element."""
  rewriter = MockRewriter()
  norm_args = [cst.Arg(value=cst.Name("x"))]
  details = {"std_args": [{"name": "inputs", "rank": 2}]}
  target_impl = {}

  result = apply_strict_guards(rewriter, norm_args, details, target_impl)
  assert len(result) == 1
  assert result[0] == norm_args[0]


def test_apply_strict_guards_arg_key_in_guards_map():
  """Test element."""
  rewriter = MockRewriter()
  norm_args = [cst.Arg(keyword=cst.Name("inputs"), value=cst.Name("x"), equal=cst.AssignEqual())]
  details = {"std_args": [{"name": "inputs", "rank": 2}]}
  target_impl = {}

  result = apply_strict_guards(rewriter, norm_args, details, target_impl)
  assert len(result) == 1
  assert isinstance(result[0].value, cst.Call)


def test_apply_strict_guards_arg_key_not_in_guards_map():
  """Test element."""
  rewriter = MockRewriter()
  norm_args = [cst.Arg(keyword=cst.Name("other"), value=cst.Name("x"), equal=cst.AssignEqual())]
  details = {"std_args": [{"name": "inputs", "rank": 2}]}
  target_impl = {}

  result = apply_strict_guards(rewriter, norm_args, details, target_impl)
  assert len(result) == 1
  assert result[0] == norm_args[0]
