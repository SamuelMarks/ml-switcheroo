"""Test suite for utils.py"""

from ml_switcheroo.enums import RuntimeTensorOrValue

from pydantic import JsonValue
from unittest.mock import MagicMock, patch

import libcst as _cst

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

cst = _cst


def parse_call(code: str) -> cst.Call:
  """Execute parse call helper logic.

  Args:
      code: Python or DSL source code string to parse or transpile.

  Returns:
      Resulting `cst.Call` from parse call.
  """
  module = cst.parse_module(code)
  expr = getattr(getattr(module.body[0], "body")[0], "value")
  return cst.ensure_type(expr, cst.Call)


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
    """Initialize test helper instance.

    Args:
        existing: The existing (list[str]) for init.
    """
    self.existing_args = existing
    self.injected_args: list[RuntimeTensorOrValue] = []


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

  def _create_dotted_name(self, name: str) -> cst.BaseExpression:
    """Execute create dotted name helper logic.

    Returns:
        Resulting `cst.BaseExpression` from create dotted name.
    """
    return cst.Name(name)


def test_rewrite_stateful_call() -> None:
  """Verify rewrite stateful call behavior."""
  rewriter = MockRewriter()
  node = parse_call("layer(x)")
  config = {"prepend_arg": "variables", "method": "apply"}

  result = rewrite_stateful_call(rewriter, node, "layer", config)
  assert getattr(
    {
      "v": getattr(
        {
          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
            getattr(
              {"v": getattr({"v": getattr({"v": result}, "__getitem__")("v").func}, "__getitem__")("v").attr},
              "__getitem__",
            )("v"),
            getattr({"v": getattr({"v": cst}, "__getitem__")("v").Name}, "__getitem__")("v"),
          )
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "apply"}, "__getitem__")("v")
  assert cst.ensure_type(result.args[0].value, cst.Name).value == "variables"

  assert len(rewriter.context.signature_stack[0].injected_args) == 1
  assert getattr(
    {
      "v": getattr(
        {
          "v": getattr(
            {
              "v": getattr(
                {
                  "v": getattr(
                    {
                      "v": getattr({"v": getattr({"v": rewriter}, "__getitem__")("v").context}, "__getitem__")(
                        "v"
                      ).signature_stack
                    },
                    "__getitem__",
                  )("v")[0]
                },
                "__getitem__",
              )("v").injected_args
            },
            "__getitem__",
          )("v")[0]
        },
        "__getitem__",
      )("v")[0]
    },
    "__getitem__",
  )("v") == getattr({"v": "variables"}, "__getitem__")("v")


def test_rewrite_stateful_call_existing() -> None:
  """Verify rewrite stateful call existing behavior."""
  rewriter = MockRewriter()
  # "variables" already exists in existing_args
  rewriter.context.signature_stack = [MockSigCtx(["variables"])]
  node = parse_call("layer(x)")
  config = {"prepend_arg": "variables"}

  rewrite_stateful_call(rewriter, node, "layer", config)
  # Shouldn't inject to signature stack
  assert len(rewriter.context.signature_stack[0].injected_args) == 0


def test_rewrite_stateful_call_no_method() -> None:
  """Verify rewrite stateful call no method behavior."""
  rewriter = MockRewriter()
  node = parse_call("layer(x)")
  config = {"prepend_arg": "vars"}

  result = rewrite_stateful_call(rewriter, node, "layer", config)
  assert cst.ensure_type(result.func, cst.Name).value == "layer"


def test_inject_kwarg() -> None:
  """Verify inject kwarg behavior."""
  node = parse_call("func(a=1)")
  result = inject_kwarg(node, "b", "b_val")
  assert len(result.args) == 2
  assert cst.ensure_type(result.args[1].keyword, cst.Name).value == "b"
  assert cst.ensure_type(result.args[1].value, cst.Name).value == "b_val"


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
  assert cst.ensure_type(result.args[0].keyword, cst.Name).value == "b"


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
  """Verify log diff behavior.

  Args:
      mock_tracer: The mock tracer (MagicMock) for test log diff.
      mock_diff: The mock diff (MagicMock) for test log diff.
  """
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

  def __init__(self, variant: dict[str, JsonValue] | None) -> None:
    """Initialize test helper instance.

    Args:
        variant: The variant (dict[str, JsonValue] | None) for init.
    """
    self.variant = variant

  def resolve_variant(self, name: str, fw: str) -> dict[str, JsonValue] | None:
    """Execute resolve variant helper logic.

    Args:
        name: Identifier or operation name string.
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue] | None` from resolve variant.
    """
    return self.variant


def test_inject_permute_call_no_api() -> None:
  """Verify inject permute call no api behavior."""
  semantics = MockSemantics({})
  node = cst.Name("x")
  result = getattr(
    {
      "v": getattr({"v": inject_permute_call}, "__getitem__")("v")(
        getattr({"v": node}, "__getitem__")("v"),
        getattr({"v": (getattr({"v": 0}, "__getitem__")("v"), getattr({"v": 1}, "__getitem__")("v"))}, "__getitem__")(
          "v"
        ),
        getattr({"v": semantics}, "__getitem__")("v"),
        getattr({"v": "jax"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert result == node


def test_inject_permute_call_kw() -> None:
  """Verify inject permute call kw behavior."""
  semantics = MockSemantics({"api": "jax.numpy.transpose", "pack_to_tuple": "axes"})
  node = cst.Name("x")
  result = getattr(
    {
      "v": getattr({"v": inject_permute_call}, "__getitem__")("v")(
        getattr({"v": node}, "__getitem__")("v"),
        getattr({"v": (getattr({"v": 1}, "__getitem__")("v"), getattr({"v": 0}, "__getitem__")("v"))}, "__getitem__")(
          "v"
        ),
        getattr({"v": semantics}, "__getitem__")("v"),
        getattr({"v": "jax"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert isinstance(result, cst.Call)
  assert getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": result}, "__getitem__")("v").func}, "__getitem__")("v").attr}, "__getitem__"
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "transpose"}, "__getitem__")("v")
  assert cst.ensure_type(result.args[1].keyword, cst.Name).value == "axes"
  assert len(cst.ensure_type(result.args[1].value, cst.Tuple).elements) == 2


def test_inject_permute_call_varargs() -> None:
  """Verify inject permute call varargs behavior."""
  semantics = MockSemantics({"api": "torch.permute"})
  node = cst.Name("x")
  result = getattr(
    {
      "v": getattr({"v": inject_permute_call}, "__getitem__")("v")(
        getattr({"v": node}, "__getitem__")("v"),
        getattr({"v": (getattr({"v": 1}, "__getitem__")("v"), getattr({"v": 0}, "__getitem__")("v"))}, "__getitem__")(
          "v"
        ),
        getattr({"v": semantics}, "__getitem__")("v"),
        getattr({"v": "torch"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v")
  assert isinstance(result, cst.Call)
  assert getattr(
    {
      "v": getattr(
        {"v": getattr({"v": getattr({"v": result}, "__getitem__")("v").func}, "__getitem__")("v").attr}, "__getitem__"
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "permute"}, "__getitem__")("v")
  assert len(result.args) == 3  # input + 2 axes
  assert cst.ensure_type(result.args[1].value, cst.Integer).value == "1"
  assert cst.ensure_type(result.args[2].value, cst.Integer).value == "0"


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
  result = rewrite_stateful_call(rewriter, node, "layer", {"prepend_arg": "v"})
  assert cst.ensure_type(result.func, cst.Name).value == "layer"


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
  result = rewrite_stateful_call(rewriter, node, "func", {})
  assert cst.ensure_type(result.func, cst.Name).value == "func"
  assert is_super_call(cst.Name("not_a_call")) is False
