"""Test suite for pre.py"""

import typing
from unittest.mock import MagicMock, patch

import libcst as cst

from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method


class MockContext:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.hook_context = MagicMock()
    self.symbol_table: typing.Any = None


class MockTraits:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.functional_execution_method = "apply"
    self.implicit_method_roots = ["torch"]


class MockSemantics:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    pass

  def get_definition(self, name: str) -> typing.Optional[tuple[str, dict[str, typing.Any]]]:
    """Docstring."""
    if name == "inplace_op":
      return ("id", {"is_inplace": True})
    return None

  def get_framework_config(self, fw: str) -> dict[str, typing.Any]:
    """Docstring."""
    return {"stateful_call": {"some": "spec"}}


class MockRewriter:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.context = MockContext()
    self.semantics = MockSemantics()
    self.target_fw = "jax"
    self.source_fw = "torch"
    self._report_warning = MagicMock()

  def _get_source_traits(self) -> MockTraits:
    return MockTraits()

  def _get_target_traits(self) -> MockTraits:
    return MockTraits()

  def _get_mapping(self, name: str, silent: bool = False) -> typing.Optional[dict[str, typing.Any]]:
    if name == "needs_plugin":
      return {"requires_plugin": True}
    if name == "Tensor.foo":
      return {"valid": True}
    if name == "torch.Tensor.foo":
      return {"valid": True}
    if name == "torch.bar":
      return {"valid": True}
    return None

  def _get_source_lifecycle_lists(self) -> tuple[set[str], set[str]]:
    return ({"strip_me"}, {"warn_me"})

  def _is_stateful(self, name: str) -> bool:
    return name == "stateful_op"

  def _is_module_alias(self, node: typing.Any) -> bool:
    return False


def parse_call(code: str) -> cst.Call:
  """Docstring."""
  module = cst.parse_module(code)
  expr = typing.cast(cst.Expr, typing.cast(cst.SimpleStatementLine, module.body[0]).body[0]).value
  return typing.cast(cst.Call, expr)


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=True)
def test_handle_pre_checks_functional_unwrap(mock_is_functional: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("layer.apply(vars, x)")
  updated = parse_call("layer.apply(vars, x)")

  handled: bool
  result: typing.Any
  handled, result = handle_pre_checks(rewriter, original, updated, "layer.apply")  # type: ignore
  assert handled
  # Result should be layer(x)
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Name)
  assert result.func.value == "layer"
  assert len(result.args) == 1
  assert typing.cast(cst.Name, result.args[0].value).value == "x"


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=True)
def test_handle_pre_checks_functional_unwrap_no_args(mock_is_functional: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("layer.apply()")
  updated = parse_call("layer.apply()")

  handled: bool
  result: typing.Any
  handled, result = handle_pre_checks(rewriter, original, updated, "layer.apply")  # type: ignore
  assert handled
  # Result should be layer()
  assert isinstance(result, cst.Call)
  assert isinstance(result.func, cst.Name)
  assert len(result.args) == 0


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=False)
@patch("ml_switcheroo.core.rewriter.calls.pre.get_hook")
def test_handle_pre_checks_unroll_inplace(mock_get_hook: MagicMock, mock_is_functional: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("inplace_op(x)")
  updated = parse_call("inplace_op(x)")

  mock_hook = MagicMock(return_value=parse_call("unrolled_op(x)"))
  mock_get_hook.return_value = mock_hook

  handled: bool
  result: typing.Any
  handled, result = handle_pre_checks(rewriter, original, updated, "inplace_op")  # type: ignore
  assert handled
  assert isinstance(result, cst.Call)
  assert typing.cast(cst.Name, result.func).value == "unrolled_op"


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=False)
def test_handle_pre_checks_lifecycle_strip(mock_is_functional: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("obj.strip_me()")
  updated = parse_call("obj.strip_me()")

  handled: bool
  result: typing.Any
  handled, result = handle_pre_checks(rewriter, original, updated, "strip_me")  # type: ignore
  assert handled
  assert isinstance(result, cst.Name)
  assert result.value == "obj"
  rewriter._report_warning.assert_called_with("Stripped framework-specific lifecycle method '.strip_me()'.")


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=False)
def test_handle_pre_checks_lifecycle_warn(mock_is_functional: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("obj.warn_me()")
  updated = parse_call("obj.warn_me()")

  handled: bool
  result: typing.Any
  handled, result = handle_pre_checks(rewriter, original, updated, "warn_me")  # type: ignore
  assert handled
  assert isinstance(result, cst.Name)
  assert result.value == "obj"
  rewriter._report_warning.assert_called_with("Ignored model state method '.warn_me()'.")


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=False)
@patch("ml_switcheroo.core.rewriter.calls.pre.rewrite_stateful_call", return_value=parse_call("new_stateful()"))
def test_handle_pre_checks_stateful(mock_rewrite: MagicMock, mock_is_functional: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("stateful_op()")
  updated = parse_call("stateful_op()")

  handled: bool
  result: typing.Any
  handled, result = handle_pre_checks(rewriter, original, updated, "stateful_op")  # type: ignore
  assert handled
  assert typing.cast(cst.Name, result.func).value == "new_stateful"


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=False)
def test_handle_pre_checks_fallback(mock_is_functional: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("normal_op()")
  updated = parse_call("normal_op()")

  handled: bool
  result: typing.Any
  handled, result = handle_pre_checks(rewriter, original, updated, "normal_op")  # type: ignore
  assert not handled
  assert result == updated


def test_handle_pre_checks_no_traits() -> None:
  """Docstring."""

  class BlankRewriter:
    def __init__(self) -> None:
      self.context = MockContext()
      self.semantics = MockSemantics()
      self.target_fw = "jax"
      self.source_fw = "torch"

      def _is_module_alias(self, n: typing.Any) -> bool:
        return False

      def _get_source_traits(self) -> typing.Any:
        """Mock."""
        return None

  rt = BlankRewriter()
  original = parse_call("layer.apply(vars, x)")
  updated = parse_call("layer.apply(vars, x)")
  handled: bool
  result: typing.Any
  handled, result = handle_pre_checks(rt, original, updated, "layer.apply")  # type: ignore
  assert not handled


def test_handle_pre_checks_plugin_claim() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("obj.strip_me()")
  updated = parse_call("obj.strip_me()")

  # Should skip strip because requires_plugin
  handled: bool
  result: typing.Any
  handled, result = handle_pre_checks(rewriter, original, updated, "needs_plugin")  # type: ignore
  assert not handled


def test_resolve_implicit_method_symbol_table() -> None:
  """Docstring."""
  rewriter = MockRewriter()

  class MockSymbolTable:
    def get_type(self, node: typing.Any) -> typing.Any:
      class MockType:
        name = "Tensor"
        framework = "torch"

      return MockType()

  rewriter.context.symbol_table = MockSymbolTable()  # type: ignore

  original = parse_call("x.foo()")
  result: typing.Any = resolve_implicit_method(rewriter, original, None)  # type: ignore
  assert result == "torch.Tensor.foo"


def test_resolve_implicit_method_legacy_fallback() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  setattr(rewriter, "source_traits", MockTraits())

  original = parse_call("someobj.bar()")
  result: typing.Any = resolve_implicit_method(rewriter, original, None)  # type: ignore
  assert result == "torch.bar"


def test_resolve_implicit_method_self() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("self.foo()")
  result: typing.Any = resolve_implicit_method(rewriter, original, None)  # type: ignore
  assert result is None


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=False)
@patch("ml_switcheroo.core.rewriter.calls.pre.get_hook")
def test_handle_pre_checks_heuristic_unroll(mock_get_hook: MagicMock, mock_is_functional: MagicMock) -> None:
  """Docstring."""
  rewriter = MockRewriter()
  original = parse_call("add_(x)")
  updated = parse_call("add_(x)")
  mock_hook = MagicMock(return_value=parse_call("new_add(x)"))
  mock_get_hook.return_value = mock_hook
  handled: bool
  result: typing.Any
  handled, result = handle_pre_checks(rewriter, original, updated, "add_")  # type: ignore
  assert handled
  assert typing.cast(cst.Name, result.func).value == "new_add"


def test_resolve_implicit_method_legacy_fallback_no_source_traits() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  rewriter.semantics.get_framework_config = MagicMock(return_value={"traits": {"implicit_method_roots": ["torch"]}})  # type: ignore
  original = parse_call("someobj.bar()")
  result: typing.Any = resolve_implicit_method(rewriter, original, None)  # type: ignore
  assert result == "torch.bar"


def test_handle_pre_checks_with_source_traits() -> None:
  """Docstring."""
  rewriter = MockRewriter()
  setattr(rewriter, "source_traits", MockTraits())
  original = parse_call("layer.apply(vars, x)")
  updated = parse_call("layer.apply(vars, x)")
  # it should hit source_traits = rewriter.source_traits
  handle_pre_checks(rewriter, original, updated, "layer.apply")  # type: ignore
