"""Test suite for the Calls Pre Dispatch module."""

from ml_switcheroo.enums import LogicOp
from pydantic import JsonValue
from unittest.mock import MagicMock, patch

import libcst as cst

from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method


class MockTraits:
  """Test helper class MockTraits."""

  def __init__(self, method: str = "apply", implicit_roots: list[str] | None = None) -> None:
    """Initializes the MockTraits instance.

    Args:
        method: Test argument or fixture.
        implicit_roots: Test argument or fixture.
    """
    self.functional_execution_method = method
    self.implicit_method_roots = implicit_roots or []


class MockType:
  """Test helper class MockType."""

  def __init__(self, name: str, framework: str | None = None) -> None:
    """Initializes the MockType instance.

    Args:
        name: Test argument or fixture.
        framework: Test argument or fixture.
    """
    self.name = name
    if framework:
      self.framework = framework


class MockSymbolTable:
  """Test helper class MockSymbolTable."""

  def __init__(self, sym_type: MockType | None = None) -> None:
    """Initializes the MockSymbolTable instance.

    Args:
        sym_type: Test argument or fixture.
    """
    self.sym_type = sym_type

  def get_type(self, node: cst.CSTNode) -> MockType | None:
    """Mock implementation of get type.

    Args:
        node: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self.sym_type


class MockContext:
  """Test helper class MockContext."""

  def __init__(self, symbol_table: MockSymbolTable | None = None) -> None:
    """Initializes the MockContext instance.

    Args:
        symbol_table: Test argument or fixture.
    """
    self.symbol_table = symbol_table
    self.hook_context = MagicMock()


class MockRewriterPre:
  """Test helper class MockRewriterPre."""

  def __init__(
    self,
    has_traits_prop: bool = False,
    has_traits_meth: bool = False,
    is_stateful_val: bool = False,
    source_framework: str = "src",
    target_framework: str = "tgt",
    is_module_val: bool = False,
    no_mapping: bool = False,
  ) -> None:
    """Initializes the MockRewriterPre instance.

    Args:
        has_traits_prop: Test argument or fixture.
        has_traits_meth: Test argument or fixture.
        is_stateful_val: Test argument or fixture.
        source_framework: Test argument or fixture.
        target_framework: Test argument or fixture.
        is_module_val: Test argument or fixture.
        no_mapping: Test argument or fixture.
    """
    if has_traits_prop:
      self.source_traits = MockTraits()
    self._is_stateful_val = is_stateful_val
    self.source_framework = source_framework
    self.target_framework = target_framework
    self.semantics = MagicMock()
    self.context = MockContext()
    self.warnings: list[str] = []
    self._is_module_val = is_module_val
    self.no_mapping = no_mapping
    if has_traits_meth:
      self._get_source_traits = lambda: MockTraits()

  def _get_mapping(self, name: str, silent: bool = True) -> dict[str, JsonValue] | None:
    """Mock implementation of  get mapping.

    Args:
        name: Test argument or fixture.
        silent: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if self.no_mapping:
      return None
    if "requires_plugin" in name:
      return {"requires_plugin": "yes"}
    if "api_found" in name:
      return {"api": name}
    return None

  def _is_stateful(self, name: str) -> bool:
    """Mock implementation of  is stateful.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._is_stateful_val

  def _report_warning(self, msg: str) -> None:
    """Mock implementation of  report warning.

    Args:
        msg: Test argument or fixture.
    """
    self.warnings.append(msg)

  def _get_source_lifecycle_lists(self) -> tuple[set[str], set[str]]:
    """Mock implementation of  get source lifecycle lists.

    Returns:
        Computed test result or fixture value.
    """
    return ({"strip_me"}, {"warn_me"})

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """Mock implementation of  is module alias.

    Args:
        node: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._is_module_val


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=True)
def test_handle_pre_checks_traits_prop(mock_is_functional: MagicMock) -> None:
  """Handles pre checks traits prop.

  Args:
      mock_is_functional: Test argument or fixture.
  """
  rewriter = MockRewriterPre(has_traits_prop=True)
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = cst.Call(
    func=cst.Attribute(value=cst.Name("layer"), attr=cst.Name("apply")),
    args=[cst.Arg(value=cst.Name("vars")), cst.Arg(value=cst.Name("x"))],
  )
  handled: bool
  handled, node = handle_pre_checks(rewriter, orig, updated, "foo")
  assert handled
  assert isinstance(getattr({"v": getattr({"v": node}, "__getitem__")("v").func}, "__getitem__")("v"), cst.Name)
  assert getattr(
    {"v": getattr({"v": getattr({"v": node}, "__getitem__")("v").func}, "__getitem__")("v").value}, "__getitem__"
  )("v") == getattr({"v": "layer"}, "__getitem__")("v")
  assert getattr(
    {"v": len(getattr({"v": getattr({"v": node}, "__getitem__")("v").args}, "__getitem__")("v"))}, "__getitem__"
  )("v") == getattr({"v": 1}, "__getitem__")("v")
  assert getattr(
    {
      "v": getattr(
        {
          "v": getattr({"v": getattr({"v": cst}, "__getitem__")("v").ensure_type}, "__getitem__")("v")(
            getattr(
              {
                "v": getattr(
                  {"v": getattr({"v": getattr({"v": node}, "__getitem__")("v").args}, "__getitem__")("v")[0]},
                  "__getitem__",
                )("v").value
              },
              "__getitem__",
            )("v"),
            getattr({"v": getattr({"v": cst}, "__getitem__")("v").Name}, "__getitem__")("v"),
          )
        },
        "__getitem__",
      )("v").value
    },
    "__getitem__",
  )("v") == getattr({"v": "x"}, "__getitem__")("v")


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=True)
def test_handle_pre_checks_traits_meth(mock_is_functional: MagicMock) -> None:
  """Handles pre checks traits meth.

  Args:
      mock_is_functional: Test argument or fixture.
  """
  rewriter = MockRewriterPre(has_traits_meth=True)
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = cst.Call(func=cst.Attribute(value=cst.Name("layer"), attr=cst.Name("apply")), args=[])
  handled: bool
  handled, node = handle_pre_checks(rewriter, orig, updated, "foo")
  assert handled
  assert getattr(
    {"v": len(getattr({"v": getattr({"v": node}, "__getitem__")("v").args}, "__getitem__")("v"))}, "__getitem__"
  )("v") == getattr({"v": 0}, "__getitem__")("v")


def test_handle_pre_checks_plugin_claim() -> None:
  """Handles pre checks plugin claim."""
  rewriter = MockRewriterPre(no_mapping=False)
  rewriter.semantics.get_definition.return_value = None
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = orig
  handled: bool
  handled, node = handle_pre_checks(rewriter, orig, updated, "requires_plugin_func")
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
    handled, node = handle_pre_checks(rewriter, orig, updated, "foo")
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
    handled, node = handle_pre_checks(rewriter, orig, updated, "foo_")
    assert handled
    assert isinstance(node, cst.Name)


def test_handle_pre_checks_lifecycle() -> None:
  """Handles pre checks lifecycle."""
  rewriter = MockRewriterPre(no_mapping=True)
  rewriter.semantics.get_definition.return_value = None
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("strip_me")), args=[])
  updated = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("strip_me")), args=[])
  handled: bool
  handled, node = handle_pre_checks(rewriter, orig, updated, "foo")
  assert handled
  assert isinstance(node, cst.Name)
  assert node.value == "obj"
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("warn_me")), args=[])
  updated = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("warn_me")), args=[])
  handled, node = handle_pre_checks(rewriter, orig, updated, "foo")
  assert handled
  assert isinstance(node, cst.Name)
  assert node.value == "obj"


@patch("ml_switcheroo.core.rewriter.calls.pre.rewrite_stateful_call", return_value=cst.Name("stateful_rewritten"))
def test_handle_pre_checks_stateful(mock_rewrite: MagicMock) -> None:
  """Handles pre checks stateful.

  Args:
      mock_rewrite: Test argument or fixture.
  """
  rewriter = MockRewriterPre(is_stateful_val=True, no_mapping=True)
  rewriter.semantics.get_definition.return_value = None
  rewriter.semantics.get_framework_config.return_value = {"stateful_call": {"method": "apply"}}
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = orig
  handled: bool
  handled, node = handle_pre_checks(rewriter, orig, updated, "foo")
  assert handled
  assert isinstance(node, cst.Name)


def test_resolve_implicit_method_self() -> None:
  """Resolves implicit method self."""
  rewriter = MockRewriterPre()
  orig = cst.Call(func=cst.Attribute(value=cst.Name("self"), attr=cst.Name("meth")), args=[])
  res = resolve_implicit_method(rewriter, orig, None)
  assert res is None


def test_resolve_implicit_method_module() -> None:
  """Resolves implicit method module."""
  rewriter = MockRewriterPre(is_module_val=True)
  orig = cst.Call(func=cst.Attribute(value=cst.Name("mod"), attr=cst.Name("meth")), args=[])
  res = resolve_implicit_method(rewriter, orig, None)
  assert res is None


def test_resolve_implicit_method_sym_table() -> None:
  """Resolves implicit method sym table."""
  rewriter = MockRewriterPre(no_mapping=False)
  rewriter.context.symbol_table = MockSymbolTable(MockType("api_found"))
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("meth")), args=[])
  res = resolve_implicit_method(rewriter, orig, None)
  assert res == "api_found.meth"
  rewriter.context.symbol_table = MockSymbolTable(MockType("Tensor", framework="fw"))
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("api_found")), args=[])
  res = resolve_implicit_method(rewriter, orig, None)
  assert res == "fw.Tensor.api_found"


def test_resolve_implicit_method_legacy_fallback() -> None:
  """Resolves implicit method legacy fallback."""
  rewriter = MockRewriterPre(no_mapping=False)
  rewriter.context.symbol_table = None
  setattr(
    getattr({"v": rewriter}, "__getitem__")("v"),
    "_get_target_traits",
    getattr({"v": getattr({"v": MagicMock}, "__getitem__")("v")()}, "__getitem__")("v"),
  )
  setattr(rewriter, "source_traits", MockTraits(implicit_roots=["api_found"]))
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("meth")), args=[])
  res = resolve_implicit_method(rewriter, orig, None)
  assert res == "api_found.meth"
  del rewriter.source_traits
  rewriter.semantics.get_framework_config.return_value = {"traits": {"implicit_method_roots": ["api_found"]}}
  res2 = resolve_implicit_method(rewriter, orig, None)
  assert res2 == "api_found.meth"


class MockRule:
  """Test helper class MockRule."""

  def __init__(
    self,
    if_arg: str,
    op: LogicOp | str,
    is_val: JsonValue | set[JsonValue] | tuple[JsonValue, ...] = None,
    use_api: str | None = None,
  ) -> None:
    """Initializes the MockRule instance.

    Args:
        if_arg: Test argument or fixture.
        op: Test argument or fixture.
        is_val: Test argument or fixture.
        use_api: Test argument or fixture.
    """
    self.if_arg = if_arg
    self.op = op
    self.is_val = is_val
    self.use_api = use_api


class MockRewriterDispatch:
  """Test helper class MockRewriterDispatch."""

  def __init__(self, source_framework: str = "src", is_module_val: bool = False) -> None:
    """Initializes the MockRewriterDispatch instance.

    Args:
        source_framework: Test argument or fixture.
        is_module_val: Test argument or fixture.
    """
    self.source_framework = source_framework
    self._is_module_val = is_module_val

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """Mock implementation of  is module alias.

    Args:
        node: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._is_module_val
