"""Unit tests for call-rewriting utility helpers (`ml_switcheroo.core.rewriter.calls.utils`)."""

from unittest.mock import MagicMock, patch

import libcst as cst
from pydantic import JsonValue

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
from ml_switcheroo.core.rewriter.types import SignatureContext
from ml_switcheroo.semantics.manager import SemanticsManager


def _parse_call(code: str) -> cst.Call:
  """Parse a Python call expression into a `cst.Call`.

  Args:
      code: Python call expression string.

  Returns:
      Parsed `cst.Call` node.
  """
  return cst.ensure_type(cst.parse_expression(code), cst.Call)


def _expr_code(node: cst.BaseExpression) -> str:
  """Serialize a CST expression node to stripped Python source code.

  Args:
      node: CST expression node.

  Returns:
      Formatted Python expression string.
  """
  return cst.Module(body=[cst.SimpleStatementLine(body=[cst.Expr(value=node)])]).code.strip()


class _StatefulContextStub:
  """Typed context stub carrying `signature_stack`."""

  def __init__(self, existing: set[str] | None = None) -> None:
    """Initialize `signature_stack` with a single `SignatureContext`.

    Args:
        existing: Existing parameter names on the enclosing function signature.
    """
    sig = SignatureContext(existing_args=existing or set())
    self.signature_stack: list[SignatureContext] = [sig]


class _StatefulRewriterStub:
  """Typed rewriter host stub supporting `rewrite_stateful_call`."""

  def __init__(self, existing: set[str] | None = None) -> None:
    """Initialize context and warning log.

    Args:
        existing: Existing parameter names in scope.
    """
    self.context = _StatefulContextStub(existing)
    self.warnings: list[str] = []

  def _create_dotted_name(self, name: str) -> cst.BaseExpression:
    """Parse `name` into a CST expression.

    Args:
        name: Dotted identifier string.

    Returns:
        Parsed `cst.BaseExpression`.
    """
    return cst.parse_expression(name)

  def _report_warning(self, msg: str) -> None:
    """Append `msg` to `self.warnings`.

    Args:
        msg: Warning message string.
    """
    self.warnings.append(msg)


def test_is_functional_apply_is_super_call_and_is_builtin() -> None:
  """Verify `is_functional_apply`, `is_super_call`, and `is_builtin` across call shapes."""
  call_apply = _parse_call("layer.apply(vars, x)")
  assert is_functional_apply(call_apply, "apply") is True
  assert is_functional_apply(call_apply, "call") is False
  assert is_functional_apply(call_apply, None) is False
  assert is_functional_apply(_parse_call("apply(x)"), "apply") is False

  assert is_super_call(_parse_call("super().__init__()")) is True
  assert is_super_call(_parse_call("super()")) is True
  assert is_super_call(_parse_call("foo.method()")) is False
  assert is_super_call(_parse_call("obj.foo().method()")) is False
  assert is_super_call(_parse_call("foo().method()")) is False
  assert is_super_call(_parse_call("funcs[0]()")) is False
  assert is_super_call(_parse_call("other()")) is False
  assert is_super_call(cst.Name("not_a_call")) is False

  assert is_builtin("len") is True
  assert is_builtin("print") is True
  assert is_builtin("custom_non_builtin") is False

  from ml_switcheroo.core.rewriter.calls.utils import is_neural_on_math_backend

  assert is_neural_on_math_backend("neural", "unknown_nonexistent_fw") is False


def test_rewrite_stateful_call_variants() -> None:
  """Verify `rewrite_stateful_call` handles `method`, `prepend_arg`, existing args, and legacy shims."""
  rewriter = _StatefulRewriterStub(existing={"x"})
  res = rewrite_stateful_call(
    rewriter,
    _parse_call("self.layer(x)"),
    "self.layer",
    {"prepend_arg": "variables", "method": "apply"},
  )
  assert _expr_code(res) == "self.layer.apply(variables, x)"
  assert rewriter.context.signature_stack[0].injected_args == [("variables", None)]
  assert len(rewriter.warnings) == 1

  # Already present in `existing_args` -> not re-injected into `signature_stack`
  existing_rewriter = _StatefulRewriterStub(existing={"variables"})
  res_existing = rewrite_stateful_call(
    existing_rewriter,
    _parse_call("layer(x)"),
    "layer",
    {"prepend_arg": "variables"},
  )
  assert _expr_code(res_existing) == "layer(variables, x)"
  assert existing_rewriter.context.signature_stack[0].injected_args == []

  # No `method` and second call on same `rewriter` (already in `injected_args` -> not duplicated)
  plain_res = rewrite_stateful_call(rewriter, _parse_call("layer(x)"), "layer", {})
  assert _expr_code(plain_res) == "layer(variables, x)"
  assert rewriter.context.signature_stack[0].injected_args == [("variables", None)]

  # Host without `context`, `_signature_stack`, or `_report_warning`
  class _ContextlessHost:
    """Minimal stateful call rewriter lacking signature stack and warning reporter."""

  ctxless_res = rewrite_stateful_call(_ContextlessHost(), _parse_call("layer(x)"), "layer", {})
  assert _expr_code(ctxless_res) == "layer(variables, x)"

  # Host with `context` but without `_report_warning`
  class _SilentHost:
    """Stateful call rewriter exposing `context` without `_report_warning`."""

    def __init__(self) -> None:
      """Initialize empty `SignatureContext`."""
      self.context = _StatefulContextStub()

  silent_res = rewrite_stateful_call(_SilentHost(), _parse_call("layer(x)"), "layer", {})
  assert _expr_code(silent_res) == "layer(variables, x)"

  # Host without `_create_dotted_name` and legacy `_signature_stack`
  class _LegacyShim:
    """Legacy rewriter stub exposing `_signature_stack` without `context` or `_create_dotted_name`."""

    def __init__(self) -> None:
      """Initialize `_signature_stack`."""
      self._signature_stack = [SignatureContext()]
      self.warned = False

    def _report_warning(self, _msg: str) -> None:
      """Mark warning flag.

      Args:
          _msg: Warning string.
      """
      self.warned = True

  shim = _LegacyShim()
  shim_res = rewrite_stateful_call(shim, _parse_call("inst(x)"), "my_inst", {"prepend_arg": "vars", "method": "apply"})
  assert _expr_code(shim_res) == "my_inst.apply(vars, x)"
  assert shim.warned is True


def test_inject_kwarg_and_strip_kwarg() -> None:
  """Verify `inject_kwarg` and `strip_kwarg` preserve existing args and clean trailing commas."""
  assert _expr_code(inject_kwarg(_parse_call("fn()"), "k", "v")) == "fn(k=v)"
  assert _expr_code(inject_kwarg(_parse_call("fn(a, b=1)"), "k", "v")) == "fn(a, b=1, k=v)"

  existing = _parse_call("fn(k=1)")
  assert inject_kwarg(existing, "k", "other") is existing

  stripped = strip_kwarg(_parse_call("fn(a=1, b=2,)"), "b")
  assert _expr_code(stripped) == "fn(a=1)"
  assert stripped.args[0].comma == cst.MaybeSentinel.DEFAULT
  assert _expr_code(strip_kwarg(_parse_call("fn(k=1)"), "k")) == "fn()"


def test_compute_permutation_and_inject_permute_call() -> None:
  """Verify `compute_permutation` and `inject_permute_call` for tuple-packed, positional, and missing APIs."""
  assert compute_permutation("NCHW", "NHWC") == (0, 2, 3, 1)
  assert compute_permutation("NC", "NCD") is None
  assert compute_permutation("NCD", "NCX") is None

  x = cst.Name("x")
  sem = MagicMock(spec=SemanticsManager)

  sem.resolve_variant.return_value = None
  assert inject_permute_call(x, (0, 1), sem, "jax") is x

  no_api: dict[str, JsonValue] = {"pack_to_tuple": "axes"}
  sem.resolve_variant.return_value = no_api
  assert inject_permute_call(x, (0, 1), sem, "jax") is x

  sem.resolve_variant.return_value = {"api": "jnp.transpose", "pack_to_tuple": "axes"}
  assert _expr_code(inject_permute_call(x, (1, 0), sem, "jax")) == "jnp.transpose(x, axes=(1, 0))"
  assert _expr_code(inject_permute_call(x, (), sem, "jax")) == "jnp.transpose(x, axes=())"

  sem.resolve_variant.return_value = {"api": "torch.permute"}
  assert _expr_code(inject_permute_call(x, (0, 2, 1), sem, "torch")) == "torch.permute(x, 0, 2, 1)"


def test_log_diff_records_only_changed_nodes() -> None:
  """Verify `log_diff` emits a mutation trace event only when `diff_nodes` detects a change."""
  with (
    patch("ml_switcheroo.core.rewriter.calls.utils.diff_nodes", return_value=("a", "b", True)),
    patch("ml_switcheroo.core.rewriter.calls.utils.get_tracer") as mock_get_tracer,
  ):
    tracer = MagicMock()
    mock_get_tracer.return_value = tracer
    log_diff("Rewrite", cst.Name("a"), cst.Name("b"))
    tracer.log_mutation.assert_called_once_with("Rewrite", "a", "b")

  with (
    patch("ml_switcheroo.core.rewriter.calls.utils.diff_nodes", return_value=("a", "a", False)),
    patch("ml_switcheroo.core.rewriter.calls.utils.get_tracer") as mock_get_tracer,
  ):
    tracer2 = MagicMock()
    mock_get_tracer.return_value = tracer2
    log_diff("NoOp", cst.Name("a"), cst.Name("a"))
    tracer2.log_mutation.assert_not_called()
