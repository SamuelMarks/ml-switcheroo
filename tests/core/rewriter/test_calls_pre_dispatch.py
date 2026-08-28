"""Test suite for the Calls Pre Dispatch module."""

import typing
import libcst as cst
from unittest.mock import MagicMock, patch
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method


class MockTraits:
  """Mock Traits class for testing purposes."""

  def __init__(self, method: str = "apply", implicit_roots: typing.Optional[list[str]] = None) -> None:
    """Initializes the MockTraits instance."""
    self.functional_execution_method = method
    self.implicit_method_roots = implicit_roots or []


class MockType:
  """Mock Type class for testing purposes."""

  def __init__(self, name: str, framework: typing.Optional[str] = None) -> None:
    """Initializes the MockType instance."""
    self.name = name
    if framework:
      self.framework = framework


class MockSymbolTable:
  """Mock Symbol Table class for testing purposes."""

  def __init__(self, sym_type: typing.Any = None) -> None:
    """Initializes the MockSymbolTable instance."""
    self.sym_type = sym_type

  def get_type(self, node: typing.Any) -> typing.Any:
    """Mock implementation of get type."""
    return self.sym_type


class MockContext:
  """Mock Context class for testing purposes."""

  def __init__(self, symbol_table: typing.Any = None) -> None:
    """Initializes the MockContext instance."""
    self.symbol_table = symbol_table
    self.hook_context = MagicMock()


class MockRewriterPre:
  """Mock Rewriter Pre class for testing purposes."""

  def __init__(
    self,
    has_traits_prop: bool = False,
    has_traits_meth: bool = False,
    is_stateful_val: bool = False,
    source_fw: str = "src",
    target_fw: str = "tgt",
    is_module_val: bool = False,
    no_mapping: bool = False,
  ) -> None:
    """Initializes the MockRewriterPre instance."""
    if has_traits_prop:
      self.source_traits = MockTraits()
    self._is_stateful_val = is_stateful_val
    self.source_fw = source_fw
    self.target_fw = target_fw
    self.semantics = MagicMock()
    self.context = MockContext()
    self.warnings: list[str] = []
    self._is_module_val = is_module_val
    self.no_mapping = no_mapping
    if has_traits_meth:
      self._get_source_traits = lambda: MockTraits()  # type: ignore

  def _get_mapping(self, name: str, silent: bool = True) -> typing.Optional[dict[str, typing.Any]]:
    """Mock implementation of  get mapping."""
    if self.no_mapping:
      return None
    if "requires_plugin" in name:
      return {"requires_plugin": "yes"}
    if "api_found" in name:
      return {"api": name}
    return None

  def _is_stateful(self, name: str) -> bool:
    """Mock implementation of  is stateful."""
    return self._is_stateful_val

  def _report_warning(self, msg: str) -> None:
    """Mock implementation of  report warning."""
    self.warnings.append(msg)

  def _get_source_lifecycle_lists(self) -> tuple[set[str], set[str]]:
    """Mock implementation of  get source lifecycle lists."""
    return ({"strip_me"}, {"warn_me"})

  def _is_module_alias(self, node: typing.Any) -> bool:
    """Mock implementation of  is module alias."""
    return self._is_module_val


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=True)
def test_handle_pre_checks_traits_prop(mock_is_functional: MagicMock) -> None:
  """Handles pre checks traits prop."""
  rewriter = MockRewriterPre(has_traits_prop=True)
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = cst.Call(
    func=cst.Attribute(value=cst.Name("layer"), attr=cst.Name("apply")),
    args=[cst.Arg(value=cst.Name("vars")), cst.Arg(value=cst.Name("x"))],
  )
  handled: bool
  node: typing.Any
  handled, node = handle_pre_checks(rewriter, orig, updated, "foo")  # type: ignore
  assert handled
  assert isinstance(node.func, cst.Name)
  assert node.func.value == "layer"
  assert len(node.args) == 1
  assert typing.cast(cst.Name, node.args[0].value).value == "x"


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=True)
def test_handle_pre_checks_traits_meth(mock_is_functional: MagicMock) -> None:
  """Handles pre checks traits meth."""
  rewriter = MockRewriterPre(has_traits_meth=True)
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = cst.Call(func=cst.Attribute(value=cst.Name("layer"), attr=cst.Name("apply")), args=[])
  handled: bool
  node: typing.Any
  handled, node = handle_pre_checks(rewriter, orig, updated, "foo")  # type: ignore
  assert handled
  assert len(node.args) == 0


def test_handle_pre_checks_plugin_claim() -> None:
  """Handles pre checks plugin claim."""
  rewriter = MockRewriterPre(no_mapping=False)
  rewriter.semantics.get_definition.return_value = None
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = orig
  handled: bool
  node: typing.Any
  handled, node = handle_pre_checks(rewriter, orig, updated, "requires_plugin_func")  # type: ignore
  assert not handled
  assert node is updated


def test_handle_pre_checks_is_inplace_and_unroll() -> None:
  """Handles pre checks is inplace and unroll."""
  rewriter = MockRewriterPre(no_mapping=True)
  rewriter.semantics.get_definition.return_value = (None, {"is_inplace": True})
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = orig
  with patch("ml_switcheroo.core.rewriter.calls.pre.get_hook") as mock_get_hook:
    mock_hook = MagicMock()
    mock_hook.return_value = cst.Name("unrolled")
    mock_get_hook.return_value = mock_hook
    handled: bool
    node: typing.Any
    handled, node = handle_pre_checks(rewriter, orig, updated, "foo")  # type: ignore
    assert handled
    assert isinstance(node, cst.Name)


def test_handle_pre_checks_endswith_underscore_unroll() -> None:
  """Handles pre checks endswith underscore unroll."""
  rewriter = MockRewriterPre(no_mapping=True)
  rewriter.semantics.get_definition.return_value = None
  orig = cst.Call(func=cst.Name("foo_"), args=[])
  updated = orig
  with patch("ml_switcheroo.core.rewriter.calls.pre.get_hook") as mock_get_hook:
    mock_hook = MagicMock()
    mock_hook.return_value = cst.Name("unrolled_")
    mock_get_hook.return_value = mock_hook
    handled: bool
    node: typing.Any
    handled, node = handle_pre_checks(rewriter, orig, updated, "foo_")  # type: ignore
    assert handled
    assert isinstance(node, cst.Name)


def test_handle_pre_checks_lifecycle() -> None:
  """Handles pre checks lifecycle."""
  rewriter = MockRewriterPre(no_mapping=True)
  rewriter.semantics.get_definition.return_value = None
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("strip_me")), args=[])
  updated = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("strip_me")), args=[])
  handled: bool
  node: typing.Any
  handled, node = handle_pre_checks(rewriter, orig, updated, "foo")  # type: ignore
  assert handled
  assert isinstance(node, cst.Name)
  assert node.value == "obj"
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("warn_me")), args=[])
  updated = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("warn_me")), args=[])
  handled, node = handle_pre_checks(rewriter, orig, updated, "foo")  # type: ignore
  assert handled
  assert isinstance(node, cst.Name)
  assert node.value == "obj"


@patch("ml_switcheroo.core.rewriter.calls.pre.rewrite_stateful_call", return_value=cst.Name("stateful_rewritten"))
def test_handle_pre_checks_stateful(mock_rewrite: MagicMock) -> None:
  """Handles pre checks stateful."""
  rewriter = MockRewriterPre(is_stateful_val=True, no_mapping=True)
  rewriter.semantics.get_definition.return_value = None
  rewriter.semantics.get_framework_config.return_value = {"stateful_call": {"method": "apply"}}
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = orig
  handled: bool
  node: typing.Any
  handled, node = handle_pre_checks(rewriter, orig, updated, "foo")  # type: ignore
  assert handled
  assert isinstance(node, cst.Name)


def test_resolve_implicit_method_self() -> None:
  """Resolves implicit method self."""
  rewriter = MockRewriterPre()
  orig = cst.Call(func=cst.Attribute(value=cst.Name("self"), attr=cst.Name("meth")), args=[])
  res: typing.Any = resolve_implicit_method(rewriter, orig, None)  # type: ignore
  assert res is None


def test_resolve_implicit_method_module() -> None:
  """Resolves implicit method module."""
  rewriter = MockRewriterPre(is_module_val=True)
  orig = cst.Call(func=cst.Attribute(value=cst.Name("mod"), attr=cst.Name("meth")), args=[])
  res: typing.Any = resolve_implicit_method(rewriter, orig, None)  # type: ignore
  assert res is None


def test_resolve_implicit_method_sym_table() -> None:
  """Resolves implicit method sym table."""
  rewriter = MockRewriterPre(no_mapping=False)
  rewriter.context.symbol_table = MockSymbolTable(MockType("api_found"))
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("meth")), args=[])
  res: typing.Any = resolve_implicit_method(rewriter, orig, None)  # type: ignore
  assert res == "api_found.meth"
  rewriter.context.symbol_table = MockSymbolTable(MockType("Tensor", framework="fw"))
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("api_found")), args=[])
  res = resolve_implicit_method(rewriter, orig, None)  # type: ignore
  assert res == "fw.Tensor.api_found"


def test_resolve_implicit_method_legacy_fallback() -> None:
  """Resolves implicit method legacy fallback."""
  rewriter = MockRewriterPre(no_mapping=False)
  rewriter.context.symbol_table = None
  rewriter._get_target_traits = MagicMock()  # type: ignore
  setattr(rewriter, "source_traits", MockTraits(implicit_roots=["api_found"]))
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("meth")), args=[])
  res: typing.Any = resolve_implicit_method(rewriter, orig, None)  # type: ignore
  assert res == "api_found.meth"
  del rewriter.source_traits  # type: ignore
  rewriter.semantics.get_framework_config.return_value = {"traits": {"implicit_method_roots": ["api_found"]}}
  res2: typing.Any = resolve_implicit_method(rewriter, orig, None)  # type: ignore
  assert res2 == "api_found.meth"


class MockRule:
  """Mock Rule class for testing purposes."""

  def __init__(self, if_arg: typing.Any, op: typing.Any, is_val: typing.Any = None, use_api: typing.Any = None) -> None:
    """Initializes the MockRule instance."""
    self.if_arg = if_arg
    self.op = op
    self.is_val = is_val
    self.use_api = use_api


class MockRewriterDispatch:
  """Mock Rewriter Dispatch class for testing purposes."""

  def __init__(self, source_fw: str = "src", is_module_val: bool = False) -> None:
    """Initializes the MockRewriterDispatch instance."""
    self.source_fw = source_fw
    self._is_module_val = is_module_val

  def _is_module_alias(self, node: typing.Any) -> bool:
    """Mock implementation of  is module alias."""
    return self._is_module_val
