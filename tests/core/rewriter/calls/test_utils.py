"""Test suite for utils.py"""

import libcst as cst
from unittest.mock import MagicMock, patch

from ml_switcheroo.core.rewriter.calls.utils import (
  is_functional_apply,
  rewrite_stateful_call,
  inject_kwarg,
  strip_kwarg,
  is_super_call,
  is_builtin,
  log_diff,
  compute_permutation,
  inject_permute_call,
)


def parse_call(code: str) -> cst.Call:
  """Docstring."""
  module = cst.parse_module(code)
  return module.body[0].body[0].value


def test_is_functional_apply():
  """Docstring."""
  call = parse_call("layer.apply(vars, x)")
  assert is_functional_apply(call, "apply")
  assert not is_functional_apply(call, "call")
  assert not is_functional_apply(parse_call("func(x)"), "apply")
  assert not is_functional_apply(call, None)


class MockSigCtx:
  """Docstring."""

  def __init__(self, existing):
    """Docstring."""
    self.existing_args = existing
    self.injected_args = []


class MockContext:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.signature_stack = [MockSigCtx(["x"])]


class MockRewriter:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.context = MockContext()
    self._report_warning = MagicMock()

  def _create_dotted_name(self, name):
    return cst.Name(name)


def test_rewrite_stateful_call():
  """Docstring."""
  rewriter = MockRewriter()
  node = parse_call("layer(x)")
  config = {"prepend_arg": "variables", "method": "apply"}

  result = rewrite_stateful_call(rewriter, node, "layer", config)
  assert result.func.attr.value == "apply"
  assert result.args[0].value.value == "variables"

  assert len(rewriter.context.signature_stack[0].injected_args) == 1
  assert rewriter.context.signature_stack[0].injected_args[0][0] == "variables"


def test_rewrite_stateful_call_existing():
  """Docstring."""
  rewriter = MockRewriter()
  # "variables" already exists in existing_args
  rewriter.context.signature_stack = [MockSigCtx(["variables"])]
  node = parse_call("layer(x)")
  config = {"prepend_arg": "variables"}

  rewrite_stateful_call(rewriter, node, "layer", config)
  # Shouldn't inject to signature stack
  assert len(rewriter.context.signature_stack[0].injected_args) == 0


def test_rewrite_stateful_call_no_method():
  """Docstring."""
  rewriter = MockRewriter()
  node = parse_call("layer(x)")
  config = {"prepend_arg": "vars"}

  result = rewrite_stateful_call(rewriter, node, "layer", config)
  assert result.func.value == "layer"


def test_inject_kwarg():
  """Docstring."""
  node = parse_call("func(a=1)")
  result = inject_kwarg(node, "b", "b_val")
  assert len(result.args) == 2
  assert result.args[1].keyword.value == "b"
  assert result.args[1].value.value == "b_val"


def test_inject_kwarg_existing():
  """Docstring."""
  node = parse_call("func(b=1)")
  result = inject_kwarg(node, "b", "b_val")
  assert result == node


def test_strip_kwarg():
  """Docstring."""
  node = parse_call("func(a=1, b=2)")
  result = strip_kwarg(node, "a")
  assert len(result.args) == 1
  assert result.args[0].keyword.value == "b"


def test_is_super_call():
  """Docstring."""
  assert is_super_call(parse_call("super().foo()"))
  assert is_super_call(parse_call("super()"))
  assert not is_super_call(parse_call("foo()"))


def test_is_builtin():
  """Docstring."""
  assert is_builtin("len")
  assert not is_builtin("my_func")


@patch("ml_switcheroo.core.rewriter.calls.utils.diff_nodes")
@patch("ml_switcheroo.core.rewriter.calls.utils.get_tracer")
def test_log_diff(mock_tracer, mock_diff):
  """Docstring."""
  mock_diff.return_value = ("a", "b", True)
  mock_tr = MagicMock()
  mock_tracer.return_value = mock_tr

  log_diff("label", cst.Pass(), cst.Pass())
  mock_tr.log_mutation.assert_called_once()

  # False case
  mock_diff.return_value = ("a", "a", False)
  mock_tr.reset_mock()
  log_diff("label", cst.Pass(), cst.Pass())
  mock_tr.log_mutation.assert_not_called()


def test_compute_permutation():
  """Docstring."""
  assert compute_permutation("NCHW", "NHWC") == (0, 2, 3, 1)
  assert compute_permutation("NC", "NCD") is None
  assert compute_permutation("NCD", "NCX") is None


class MockSemantics:
  """Docstring."""

  def __init__(self, variant):
    """Docstring."""
    self.variant = variant

  def resolve_variant(self, name, fw):
    """Docstring."""
    return self.variant


def test_inject_permute_call_no_api():
  """Docstring."""
  semantics = MockSemantics({})
  node = cst.Name("x")
  result = inject_permute_call(node, (0, 1), semantics, "jax")
  assert result == node


def test_inject_permute_call_kw():
  """Docstring."""
  semantics = MockSemantics({"api": "jax.numpy.transpose", "pack_to_tuple": "axes"})
  node = cst.Name("x")
  result = inject_permute_call(node, (1, 0), semantics, "jax")
  assert isinstance(result, cst.Call)
  assert result.func.attr.value == "transpose"
  assert result.args[1].keyword.value == "axes"
  assert len(result.args[1].value.elements) == 2


def test_inject_permute_call_varargs():
  """Docstring."""
  semantics = MockSemantics({"api": "torch.permute"})
  node = cst.Name("x")
  result = inject_permute_call(node, (1, 0), semantics, "torch")
  assert isinstance(result, cst.Call)
  assert result.func.attr.value == "permute"
  assert len(result.args) == 3  # input + 2 axes
  assert result.args[1].value.value == "1"
  assert result.args[2].value.value == "0"


def test_rewrite_stateful_call_legacy():
  """Docstring."""

  class LegacyRewriter:
    def __init__(self):
      self._signature_stack = [MockSigCtx(["x"])]
      self._report_warning = MagicMock()

  rewriter = LegacyRewriter()
  node = parse_call("layer(x)")
  result = rewrite_stateful_call(rewriter, node, "layer", {"prepend_arg": "v"})
  assert result.func.value == "layer"


def test_strip_kwarg_trailing_comma():
  """Docstring."""
  node = parse_call("func(a=1, b=2,)")
  result = strip_kwarg(node, "b")
  assert len(result.args) == 1
  assert result.args[0].comma == cst.MaybeSentinel.DEFAULT


def test_rewrite_stateful_call_no_method_coverage():
  """Docstring."""
  rewriter = MockRewriter()
  node = cst.Call(func=cst.Name("func"), args=[])
  result = rewrite_stateful_call(rewriter, node, "func", {})
  assert result.func.value == "func"
