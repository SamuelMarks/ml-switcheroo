"""Test suite for pre.py"""

import libcst as cst
from unittest.mock import MagicMock, patch

from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method


class MockContext:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.hook_context = MagicMock()
    self.symbol_table = None


class MockTraits:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.functional_execution_method = "apply"
    self.implicit_method_roots = ["torch"]


class MockSemantics:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    pass

  def get_definition(self, name):
    """Docstring."""
    if name == "inplace_op":
      return ("id", {"is_inplace": True})
    return None

  def get_framework_config(self, fw):
    """Docstring."""
    return {"stateful_call": {"some": "spec"}}


class MockRewriter:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.context = MockContext()
    self.semantics = MockSemantics()
    self.target_fw = "jax"
    self.source_fw = "torch"
    self._report_warning = MagicMock()

  def _get_source_traits(self):
    return MockTraits()

  def _get_target_traits(self):
    return MockTraits()

  def _get_mapping(self, name, silent=False):
    if name == "needs_plugin":
      return {"requires_plugin": True}
    if name == "Tensor.foo":
      return {"valid": True}
    if name == "torch.Tensor.foo":
      return {"valid": True}
    if name == "torch.bar":
      return {"valid": True}
    return None

  def _get_source_lifecycle_lists(self):
    return ({"strip_me"}, {"warn_me"})

  def _is_stateful(self, name):
    return name == "stateful_op"

  def _is_module_alias(self, node):
    return False


def parse_call(code: str) -> cst.Call:
  """Docstring."""
  module = cst.parse_module(code)
  return module.body[0].body[0].value


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=True)
def test_handle_pre_checks_functional_unwrap(mock_is_functional):
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("layer.apply(vars, x)")
  updated = parse_call("layer.apply(vars, x)")

  handled, result = handle_pre_checks(rewriter, original, updated, "layer.apply")
  assert handled
  # Result should be layer(x)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Name)
  assert result.func.value == "layer"
  assert len(result.args) == 1
  assert result.args[0].value.value == "x"


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=True)
def test_handle_pre_checks_functional_unwrap_no_args(mock_is_functional):
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("layer.apply()")
  updated = parse_call("layer.apply()")

  handled, result = handle_pre_checks(rewriter, original, updated, "layer.apply")
  assert handled
  # Result should be layer()
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Name)
  assert len(result.args) == 0


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=False)
@patch("ml_switcheroo.core.rewriter.calls.pre.get_hook")
def test_handle_pre_checks_unroll_inplace(mock_get_hook, mock_is_functional):
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("inplace_op(x)")
  updated = parse_call("inplace_op(x)")

  mock_hook = MagicMock(return_value=parse_call("unrolled_op(x)"))
  mock_get_hook.return_value = mock_hook

  handled, result = handle_pre_checks(rewriter, original, updated, "inplace_op")
  assert handled
  assert isinstance(result, cst.Call)
  assert result.func.value == "unrolled_op"


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=False)
def test_handle_pre_checks_lifecycle_strip(mock_is_functional):
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("obj.strip_me()")
  updated = parse_call("obj.strip_me()")

  handled, result = handle_pre_checks(rewriter, original, updated, "strip_me")
  assert handled
  assert isinstance(result, cst.Name)
  assert result.value == "obj"
  rewriter._report_warning.assert_called_with("Stripped framework-specific lifecycle method '.strip_me()'.")


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=False)
def test_handle_pre_checks_lifecycle_warn(mock_is_functional):
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("obj.warn_me()")
  updated = parse_call("obj.warn_me()")

  handled, result = handle_pre_checks(rewriter, original, updated, "warn_me")
  assert handled
  assert isinstance(result, cst.Name)
  assert result.value == "obj"
  rewriter._report_warning.assert_called_with("Ignored model state method '.warn_me()'.")


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=False)
@patch("ml_switcheroo.core.rewriter.calls.pre.rewrite_stateful_call", return_value=parse_call("new_stateful()"))
def test_handle_pre_checks_stateful(mock_rewrite, mock_is_functional):
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("stateful_op()")
  updated = parse_call("stateful_op()")

  handled, result = handle_pre_checks(rewriter, original, updated, "stateful_op")
  assert handled
  assert result.func.value == "new_stateful"


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=False)
def test_handle_pre_checks_fallback(mock_is_functional):
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("normal_op()")
  updated = parse_call("normal_op()")

  handled, result = handle_pre_checks(rewriter, original, updated, "normal_op")
  assert not handled
  assert result == updated


def test_handle_pre_checks_no_traits():
  """Docstring."""

  class BlankRewriter:
    def __init__(self):
      self.context = MockContext()
      self.semantics = MockSemantics()
      self.target_fw = "jax"
      self.source_fw = "torch"

    def _is_module_alias(self, n):
      return False

  rt = BlankRewriter()
  original = parse_call("layer.apply(vars, x)")
  updated = parse_call("layer.apply(vars, x)")
  handled, result = handle_pre_checks(rt, original, updated, "layer.apply")
  assert not handled


def test_handle_pre_checks_plugin_claim():
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("obj.strip_me()")
  updated = parse_call("obj.strip_me()")

  # Should skip strip because requires_plugin
  handled, result = handle_pre_checks(rewriter, original, updated, "needs_plugin")
  assert not handled


def test_resolve_implicit_method_symbol_table():
  """Docstring."""
  rewriter = MockRewriter()

  class MockSymbolTable:
    def get_type(self, node):
      class MockType:
        name = "Tensor"
        framework = "torch"

      return MockType()

  rewriter.context.symbol_table = MockSymbolTable()

  original = parse_call("x.foo()")
  result = resolve_implicit_method(rewriter, original, None)
  assert result == "torch.Tensor.foo"


def test_resolve_implicit_method_legacy_fallback():
  """Docstring."""
  rewriter = MockRewriter()
  rewriter.source_traits = MockTraits()

  original = parse_call("someobj.bar()")
  result = resolve_implicit_method(rewriter, original, None)
  assert result == "torch.bar"


def test_resolve_implicit_method_self():
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("self.foo()")
  result = resolve_implicit_method(rewriter, original, None)
  assert result is None


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=False)
@patch("ml_switcheroo.core.rewriter.calls.pre.get_hook")
def test_handle_pre_checks_heuristic_unroll(mock_get_hook, mock_is_functional):
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("add_(x)")
  updated = parse_call("add_(x)")
  mock_hook = MagicMock(return_value=parse_call("new_add(x)"))
  mock_get_hook.return_value = mock_hook
  handled, result = handle_pre_checks(rewriter, original, updated, "add_")
  assert handled
  assert result.func.value == "new_add"


def test_resolve_implicit_method_legacy_fallback_no_source_traits():
  """Docstring."""
  rewriter = MockRewriter()
  rewriter.semantics.get_framework_config = MagicMock(return_value={"traits": {"implicit_method_roots": ["torch"]}})
  original = parse_call("someobj.bar()")
  result = resolve_implicit_method(rewriter, original, None)
  assert result == "torch.bar"


def test_handle_pre_checks_with_source_traits():
  """Docstring."""
  rewriter = MockRewriter()
  rewriter.source_traits = MockTraits()
  original = parse_call("layer.apply(vars, x)")
  updated = parse_call("layer.apply(vars, x)")
  # it should hit source_traits = rewriter.source_traits
  handle_pre_checks(rewriter, original, updated, "layer.apply")
