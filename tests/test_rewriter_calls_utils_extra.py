"""Tests for ml_switcheroo.core.rewriter.calls.utils."""

import libcst as cst
from unittest.mock import MagicMock
from ml_switcheroo.core.rewriter.calls.utils import (
  is_functional_apply,
  rewrite_stateful_call,
  inject_kwarg,
  strip_kwarg,
  is_super_call,
  is_builtin,
  compute_permutation,
  inject_permute_call,
)


def test_is_functional_apply_missing_method_name():
  """Test is_functional_apply with missing method name returns False."""
  call_node = cst.parse_expression("foo.apply()")
  assert is_functional_apply(call_node, None) is False


def test_is_functional_apply_not_attribute():
  """Test is_functional_apply with non-attribute func."""
  call_node = cst.parse_expression("apply()")
  assert is_functional_apply(call_node) is False


class RewriterShim:
  """Test element."""

  def __init__(self, stack):
    """Test element."""
    self._signature_stack = stack
    self.warned = False

  def _report_warning(self, msg):
    self.warned = True


def test_rewrite_stateful_call_legacy_shim():
  """Test rewrite_stateful_call with legacy _signature_stack shim."""
  sig_ctx = MagicMock()
  sig_ctx.existing_args = []
  sig_ctx.injected_args = []
  rewriter = RewriterShim([sig_ctx])

  node = cst.parse_expression("instance()")
  new_node = rewrite_stateful_call(rewriter, node, "instance", {"prepend_arg": "variables"})

  assert len(new_node.args) == 1
  assert rewriter.warned


def test_rewrite_stateful_call_no_method():
  """Test rewrite_stateful_call without a configured method name."""
  rewriter = MagicMock()
  rewriter.context.signature_stack = []
  node = cst.parse_expression("instance()")
  new_node = rewrite_stateful_call(rewriter, node, "instance", {})
  assert new_node.func.value == "instance"


def test_rewrite_stateful_call_create_dotted_name_fallback():
  """Test rewrite_stateful_call falling back when _create_dotted_name is missing."""
  rewriter = MagicMock()
  rewriter.context.signature_stack = []
  del rewriter._create_dotted_name
  node = cst.parse_expression("instance()")
  new_node = rewrite_stateful_call(rewriter, node, "instance", {"method": "apply"})
  assert isinstance(new_node.func, cst.Attribute)
  assert new_node.func.value.value == "instance"


def test_inject_kwarg_existing():
  """Test inject_kwarg when argument already exists."""
  node = cst.parse_expression("foo(x=1)")
  new_node = inject_kwarg(node, "x", "two")
  assert len(new_node.args) == 1
  assert new_node.args[0].value.value == "1"


def test_inject_kwarg_existing_positional():
  """Test inject_kwarg when adding to existing positional args."""
  node = cst.parse_expression("foo(y)")
  new_node = inject_kwarg(node, "x", "two")
  assert len(new_node.args) == 2


def test_strip_kwarg_trailing_comma_cleanup():
  """Test strip_kwarg removes trailing comma on new last arg."""
  node = cst.parse_expression("foo(a=1, x=2)")
  new_node = strip_kwarg(node, "x")
  assert len(new_node.args) == 1
  assert new_node.args[0].comma == cst.MaybeSentinel.DEFAULT


def test_is_super_call_other_calls():
  """Test is_super_call branches."""
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


def test_is_builtin_extra():
  """Test is_builtin for coverage."""
  assert is_builtin("len")
  assert not is_builtin("not_a_builtin")


def test_compute_permutation():
  """Test compute_permutation."""
  # Invalid lengths
  assert compute_permutation("NCHW", "NHW") is None
  # Character not found
  assert compute_permutation("NCHW", "NHWX") is None
  # Valid
  assert compute_permutation("NCHW", "NHWC") == (0, 2, 3, 1)


def test_inject_permute_call_no_variant():
  """Test inject_permute_call where no semantics variant is found."""
  semantics = MagicMock()
  semantics.resolve_variant.return_value = None
  node = cst.parse_expression("x")
  new_node = inject_permute_call(node, (0, 1), semantics, "torch")
  assert new_node == node


def test_inject_permute_call_no_api_in_variant():
  """Test inject_permute_call where variant has no API string."""
  semantics = MagicMock()
  semantics.resolve_variant.return_value = {"pack_to_tuple": "dim"}
  node = cst.parse_expression("x")
  new_node = inject_permute_call(node, (0, 1), semantics, "torch")
  assert new_node == node


def test_inject_permute_call_multi_part_api():
  """Test inject_permute_call with multi-part API."""
  semantics = MagicMock()
  semantics.resolve_variant.return_value = {"api": "tf.transpose"}
  node = cst.parse_expression("x")
  new_node = inject_permute_call(node, (0, 1), semantics, "tf")
  assert isinstance(new_node.func, cst.Attribute)


def test_inject_permute_call_positional_args():
  """Test inject_permute_call generating positional args."""
  semantics = MagicMock()
  semantics.resolve_variant.return_value = {"api": "torch.permute"}
  node = cst.parse_expression("x")
  new_node = inject_permute_call(node, (0, 2, 1), semantics, "torch")
  assert len(new_node.args) == 4
  assert new_node.args[1].value.value == "0"
  assert new_node.args[2].value.value == "2"
  assert new_node.args[3].value.value == "1"


def test_inject_permute_call_pack_kw_with_elements():
  """Test inject_permute_call with pack_kw and non-empty indices."""
  semantics = MagicMock()
  semantics.resolve_variant.return_value = {"api": "torch.transpose", "pack_to_tuple": "dim"}
  node = cst.parse_expression("x")
  new_node = inject_permute_call(node, (0, 2, 1), semantics, "torch")
  assert len(new_node.args) == 2
  kwarg = new_node.args[1]
  assert kwarg.keyword.value == "dim"
  assert isinstance(kwarg.value, cst.Tuple)
  assert len(kwarg.value.elements) == 3


def test_rewrite_stateful_call_with_dotted_name():
  """Test rewrite_stateful_call when _create_dotted_name is available."""
  rewriter = MagicMock()
  rewriter.context.signature_stack = []
  rewriter._create_dotted_name.return_value = cst.Name("my_instance")
  node = cst.parse_expression("instance()")
  new_node = rewrite_stateful_call(rewriter, node, "instance", {"method": "apply"})
  assert isinstance(new_node.func, cst.Attribute)
  assert new_node.func.value.value == "my_instance"
