"""Tests for ml_switcheroo.core.rewriter.calls.utils."""

from ml_switcheroo.core.rewriter.types import SignatureContext

from unittest.mock import MagicMock

import libcst as cst

from ml_switcheroo.core.rewriter.calls.utils import (
  compute_permutation,
  inject_kwarg,
  inject_permute_call,
  is_builtin,
  is_functional_apply,
  is_super_call,
  rewrite_stateful_call,
  strip_kwarg,
)


def test_is_functional_apply_missing_method_name() -> None:
  """Verify is functional apply missing method name behavior."""
  call_node = cst.ensure_type(cst.parse_expression("foo.apply()"), cst.Call)
  assert is_functional_apply(call_node, None) is False


def test_is_functional_apply_not_attribute() -> None:
  """Verify is functional apply not attribute behavior."""
  call_node = cst.ensure_type(cst.parse_expression("apply()"), cst.Call)
  assert is_functional_apply(call_node) is False


class RewriterShim:
  """Test helper class RewriterShim."""

  def __init__(self, stack: list[SignatureContext]) -> None:
    """Initialize test helper instance.

    Args:
        stack: The stack (list[SignatureContext]) for   init  .
    """
    self._signature_stack = stack
    self.warned = False

  def _report_warning(self, msg: str) -> None:
    """Execute report warning helper logic.

    Args:
        msg: Warning or error message string.
    """
    self.warned = True


def test_rewrite_stateful_call_legacy_shim() -> None:
  """Verify rewrite stateful call legacy shim behavior."""
  sig_ctx = MagicMock()
  sig_ctx.existing_args = []
  sig_ctx.injected_args = []
  rewriter = RewriterShim([sig_ctx])

  node = cst.ensure_type(cst.parse_expression("instance()"), cst.Call)
  new_node = rewrite_stateful_call(rewriter, node, "instance", {"prepend_arg": "variables"})

  assert len(new_node.args) == 1
  assert rewriter.warned


def test_rewrite_stateful_call_no_method() -> None:
  """Verify rewrite stateful call no method behavior."""
  rewriter = MagicMock()
  rewriter.context.signature_stack = []
  node = cst.ensure_type(cst.parse_expression("instance()"), cst.Call)
  new_node = rewrite_stateful_call(rewriter, node, "instance", {})
  assert getattr(new_node.func, "value", None) == "instance"


def test_rewrite_stateful_call_create_dotted_name_fallback() -> None:
  """Verify rewrite stateful call create dotted name fallback behavior."""
  rewriter = MagicMock()
  rewriter.context.signature_stack = []
  del rewriter._create_dotted_name
  node = cst.ensure_type(cst.parse_expression("instance()"), cst.Call)
  new_node = rewrite_stateful_call(rewriter, node, "instance", {"method": "apply"})
  assert isinstance(new_node.func, cst.Attribute)
  assert getattr(new_node.func.value, "value", None) == "instance"


def test_inject_kwarg_existing() -> None:
  """Verify inject kwarg existing behavior."""
  node = cst.ensure_type(cst.parse_expression("foo(x=1)"), cst.Call)
  new_node = inject_kwarg(node, "x", "two")
  assert len(new_node.args) == 1
  assert getattr(new_node.args[0].value, "value", None) == "1"


def test_inject_kwarg_existing_positional() -> None:
  """Verify inject kwarg existing positional behavior."""
  node = cst.ensure_type(cst.parse_expression("foo(y)"), cst.Call)
  new_node = inject_kwarg(node, "x", "two")
  assert len(new_node.args) == 2


def test_strip_kwarg_trailing_comma_cleanup() -> None:
  """Verify strip kwarg trailing comma cleanup behavior."""
  node = cst.ensure_type(cst.parse_expression("foo(a=1, x=2)"), cst.Call)
  new_node = strip_kwarg(node, "x")
  assert len(new_node.args) == 1
  assert new_node.args[0].comma == cst.MaybeSentinel.DEFAULT


def test_is_super_call_other_calls() -> None:
  """Verify is super call other calls behavior."""
  # Attribute but receiver is not a call
  assert not is_super_call(cst.parse_expression("foo.method()"))
  # Attribute but receiver's func is not a Name
  assert not is_super_call(cst.parse_expression("obj.foo().method()"))
  # Attribute where call is not super
  assert not is_super_call(cst.parse_expression("foo().method()"))
  # Just a name call that is not super
  assert not is_super_call(cst.parse_expression("not_super()"))

  # Test the True cases
  assert is_super_call(cst.parse_expression("super().method()"))
  assert is_super_call(cst.parse_expression("super()"))


def test_is_builtin_extra() -> None:
  """Verify is builtin extra behavior."""
  assert is_builtin("len")
  assert not is_builtin("not_a_builtin")


def test_compute_permutation() -> None:
  """Verify compute permutation behavior."""
  # Invalid lengths
  assert compute_permutation("NCHW", "NHW") is None
  # Character not found
  assert compute_permutation("NCHW", "NHWX") is None
  # Valid
  assert compute_permutation("NCHW", "NHWC") == (0, 2, 3, 1)


def test_inject_permute_call_no_variant() -> None:
  """Verify inject permute call no variant behavior."""
  semantics = MagicMock()
  semantics.resolve_variant.return_value = None
  node = cst.parse_expression("x")
  new_node = inject_permute_call(node, (0, 1), semantics, "torch")
  assert new_node == node


def test_inject_permute_call_no_api_in_variant() -> None:
  """Verify inject permute call no api in variant behavior."""
  semantics = MagicMock()
  semantics.resolve_variant.return_value = {"pack_to_tuple": "dim"}
  node = cst.parse_expression("x")
  new_node = inject_permute_call(node, (0, 1), semantics, "torch")
  assert new_node == node


def test_inject_permute_call_multi_part_api() -> None:
  """Verify inject permute call multi part api behavior."""
  semantics = MagicMock()
  semantics.resolve_variant.return_value = {"api": "tf.transpose"}
  node = cst.parse_expression("x")
  new_node = inject_permute_call(node, (0, 1), semantics, "tf")
  assert isinstance(new_node, cst.Call)
  assert isinstance(new_node.func, cst.Attribute)


def test_inject_permute_call_positional_args() -> None:
  """Verify inject permute call positional args behavior."""
  semantics = MagicMock()
  semantics.resolve_variant.return_value = {"api": "torch.permute"}
  node = cst.parse_expression("x")
  new_node = inject_permute_call(node, (0, 2, 1), semantics, "torch")
  assert isinstance(new_node, cst.Call)
  assert len(new_node.args) == 4
  assert getattr(new_node.args[1].value, "value", None) == "0"
  assert getattr(new_node.args[2].value, "value", None) == "2"
  assert getattr(new_node.args[3].value, "value", None) == "1"


def test_inject_permute_call_pack_kw_with_elements() -> None:
  """Verify inject permute call pack kw with elements behavior."""
  semantics = MagicMock()
  semantics.resolve_variant.return_value = {"api": "torch.transpose", "pack_to_tuple": "dim"}
  node = cst.parse_expression("x")
  new_node = inject_permute_call(node, (0, 2, 1), semantics, "torch")
  assert isinstance(new_node, cst.Call)
  assert len(new_node.args) == 2
  kwarg = new_node.args[1]
  assert getattr(kwarg.keyword, "value", None) == "dim"
  assert isinstance(kwarg.value, cst.Tuple)
  assert len(kwarg.value.elements) == 3


def test_rewrite_stateful_call_with_dotted_name() -> None:
  """Verify rewrite stateful call with dotted name behavior."""
  rewriter = MagicMock()
  rewriter.context.signature_stack = []
  rewriter._create_dotted_name.return_value = cst.Name("my_instance")
  node = cst.ensure_type(cst.parse_expression("instance()"), cst.Call)
  new_node = rewrite_stateful_call(rewriter, node, "instance", {"method": "apply"})
  assert isinstance(new_node.func, cst.Attribute)
  assert getattr(new_node.func.value, "value", None) == "my_instance"
