"""Test suite for utils.py"""

import typing
from unittest.mock import MagicMock, patch

import libcst as cst

from ml_switcheroo.core.rewriter.calls.utils import (
  compute_permutation,
  inject_kwarg,
  inject_permute_call,
  is_builtin,
  is_functional_apply,
  is_super_call,
  log_diff,
  rewrite_stateful_call,
  strip_kwarg,
)


def parse_call(code: str) -> cst.Call:
  """Execute parse call helper logic."""
  module = cst.parse_module(code)
  expr = typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]).value
  return typing.cast(cst.Call, expr)


def test_is_functional_apply() -> None:
  """Verify is functional apply behavior."""
  call = parse_call("layer.apply(vars, x)")
  assert is_functional_apply(call, "apply")
  assert not is_functional_apply(call, "call")
  assert not is_functional_apply(parse_call("func(x)"), "apply")
  assert not is_functional_apply(call, None)


class MockSigCtx:
  """Test helper class MockSigCtx."""

  def __init__(self, existing: list[str]) -> None:
    """Initialize test helper instance."""
    self.existing_args = existing
    self.injected_args: list[tuple[str, typing.Optional[cst.CSTNode]]] = []


class MockContext:
  """Test helper class MockContext."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.signature_stack = [MockSigCtx(["x"])]


class MockRewriter:
  """Test helper class MockRewriter."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.context = MockContext()
    self._report_warning = MagicMock()

  def _create_dotted_name(self, name: str) -> cst.Name:
    """Execute create dotted name helper logic."""
    return cst.Name(name)


def test_rewrite_stateful_call() -> None:
  """Verify rewrite stateful call behavior."""
  rewriter = MockRewriter()
  node = parse_call("layer(x)")
  config: dict[str, typing.Any] = {"prepend_arg": "variables", "method": "apply"}

  result: typing.Any = rewrite_stateful_call(rewriter, node, "layer", config)  # type: ignore
  assert typing.cast(cst.Name, result.func.attr).value == "apply"
  assert typing.cast(cst.Name, result.args[0].value).value == "variables"

  assert len(rewriter.context.signature_stack[0].injected_args) == 1
  assert rewriter.context.signature_stack[0].injected_args[0][0] == "variables"


def test_rewrite_stateful_call_existing() -> None:
  """Verify rewrite stateful call existing behavior."""
  rewriter = MockRewriter()
  # "variables" already exists in existing_args
  rewriter.context.signature_stack = [MockSigCtx(["variables"])]
  node = parse_call("layer(x)")
  config: dict[str, typing.Any] = {"prepend_arg": "variables"}

  rewrite_stateful_call(rewriter, node, "layer", config)  # type: ignore
  # Shouldn't inject to signature stack
  assert len(rewriter.context.signature_stack[0].injected_args) == 0


def test_rewrite_stateful_call_no_method() -> None:
  """Verify rewrite stateful call no method behavior."""
  rewriter = MockRewriter()
  node = parse_call("layer(x)")
  config: dict[str, typing.Any] = {"prepend_arg": "vars"}

  result: typing.Any = rewrite_stateful_call(rewriter, node, "layer", config)  # type: ignore
  assert typing.cast(cst.Name, result.func).value == "layer"


def test_inject_kwarg() -> None:
  """Verify inject kwarg behavior."""
  node = parse_call("func(a=1)")
  result = inject_kwarg(node, "b", "b_val")
  assert len(result.args) == 2
  assert typing.cast(cst.Name, result.args[1].keyword).value == "b"
  assert typing.cast(cst.Name, result.args[1].value).value == "b_val"


def test_inject_kwarg_existing() -> None:
  """Verify inject kwarg existing behavior."""
  node = parse_call("func(b=1)")
  result = inject_kwarg(node, "b", "b_val")
  assert result == node


def test_strip_kwarg() -> None:
  """Verify strip kwarg behavior."""
  node = parse_call("func(a=1, b=2)")
  result = strip_kwarg(node, "a")
  assert len(result.args) == 1
  assert typing.cast(cst.Name, result.args[0].keyword).value == "b"


def test_is_super_call() -> None:
  """Verify is super call behavior."""
  assert is_super_call(parse_call("super().foo()"))
  assert is_super_call(parse_call("super()"))
  assert not is_super_call(parse_call("foo()"))


def test_is_builtin() -> None:
  """Verify is builtin behavior."""
  assert is_builtin("len")
  assert not is_builtin("my_func")


@patch("ml_switcheroo.core.rewriter.calls.utils.diff_nodes")
@patch("ml_switcheroo.core.rewriter.calls.utils.get_tracer")
def test_log_diff(mock_tracer: MagicMock, mock_diff: MagicMock) -> None:
  """Verify log diff behavior."""
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


def test_compute_permutation() -> None:
  """Verify compute permutation behavior."""
  assert compute_permutation("NCHW", "NHWC") == (0, 2, 3, 1)
  assert compute_permutation("NC", "NCD") is None
  assert compute_permutation("NCD", "NCX") is None


class MockSemantics:
  """Test helper class MockSemantics."""

  def __init__(self, variant: typing.Any) -> None:
    """Initialize test helper instance."""
    self.variant = variant

  def resolve_variant(self, name: str, fw: str) -> typing.Any:
    """Execute resolve variant helper logic."""
    return self.variant


def test_inject_permute_call_no_api() -> None:
  """Verify inject permute call no api behavior."""
  semantics = MockSemantics({})
  node = cst.Name("x")
  result: typing.Any = inject_permute_call(node, (0, 1), semantics, "jax")  # type: ignore
  assert result == node


def test_inject_permute_call_kw() -> None:
  """Verify inject permute call kw behavior."""
  semantics = MockSemantics({"api": "jax.numpy.transpose", "pack_to_tuple": "axes"})
  node = cst.Name("x")
  result: typing.Any = inject_permute_call(node, (1, 0), semantics, "jax")  # type: ignore
  assert isinstance(result, cst.Call)
  assert typing.cast(cst.Name, typing.cast(cst.Attribute, result.func).attr).value == "transpose"
  assert typing.cast(cst.Name, result.args[1].keyword).value == "axes"
  assert len(typing.cast(cst.Tuple, result.args[1].value).elements) == 2


def test_inject_permute_call_varargs() -> None:
  """Verify inject permute call varargs behavior."""
  semantics = MockSemantics({"api": "torch.permute"})
  node = cst.Name("x")
  result: typing.Any = inject_permute_call(node, (1, 0), semantics, "torch")  # type: ignore
  assert isinstance(result, cst.Call)
  assert typing.cast(cst.Name, typing.cast(cst.Attribute, result.func).attr).value == "permute"
  assert len(result.args) == 3  # input + 2 axes
  assert typing.cast(cst.Integer, result.args[1].value).value == "1"
  assert typing.cast(cst.Integer, result.args[2].value).value == "0"


def test_rewrite_stateful_call_legacy() -> None:
  """Verify rewrite stateful call legacy behavior."""

  class LegacyRewriter:
    """Test helper class LegacyRewriter."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self._signature_stack = [MockSigCtx(["x"])]
      self._report_warning = MagicMock()

  rewriter = LegacyRewriter()
  node = parse_call("layer(x)")
  result: typing.Any = rewrite_stateful_call(rewriter, node, "layer", {"prepend_arg": "v"})  # type: ignore
  assert typing.cast(cst.Name, result.func).value == "layer"


def test_strip_kwarg_trailing_comma() -> None:
  """Verify strip kwarg trailing comma behavior."""
  node = parse_call("func(a=1, b=2,)")
  result = strip_kwarg(node, "b")
  assert len(result.args) == 1
  assert result.args[0].comma == cst.MaybeSentinel.DEFAULT


def test_rewrite_stateful_call_no_method_coverage() -> None:
  """Verify rewrite stateful call no method coverage behavior."""
  rewriter = MockRewriter()
  node = cst.Call(func=cst.Name("func"), args=[])
  result: typing.Any = rewrite_stateful_call(rewriter, node, "func", {})  # type: ignore
  assert typing.cast(cst.Name, result.func).value == "func"
