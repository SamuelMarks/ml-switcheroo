"""Test suite for the Calls Pre Dispatch module."""

import libcst as cst
from unittest.mock import MagicMock, patch
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method


class MockTraits:
  """Mock Traits class for testing purposes."""

  def __init__(self, method="apply", implicit_roots=None):
    """Initializes the MockTraits instance."""
    self.functional_execution_method = method
    self.implicit_method_roots = implicit_roots or []


class MockType:
  """Mock Type class for testing purposes."""

  def __init__(self, name, framework=None):
    """Initializes the MockType instance."""
    self.name = name
    if framework:
      self.framework = framework


class MockSymbolTable:
  """Mock Symbol Table class for testing purposes."""

  def __init__(self, sym_type=None):
    """Initializes the MockSymbolTable instance."""
    self.sym_type = sym_type

  def get_type(self, node):
    """Mock implementation of get type."""
    return self.sym_type


class MockContext:
  """Mock Context class for testing purposes."""

  def __init__(self, symbol_table=None):
    """Initializes the MockContext instance."""
    self.symbol_table = symbol_table
    self.hook_context = MagicMock()


class MockRewriterPre:
  """Mock Rewriter Pre class for testing purposes."""

  def __init__(
    self,
    has_traits_prop=False,
    has_traits_meth=False,
    is_stateful_val=False,
    source_fw="src",
    target_fw="tgt",
    is_module_val=False,
    no_mapping=False,
  ):
    """Initializes the MockRewriterPre instance."""
    if has_traits_prop:
      self.source_traits = MockTraits()
    self._is_stateful_val = is_stateful_val
    self.source_fw = source_fw
    self.target_fw = target_fw
    self.semantics = MagicMock()
    self.context = MockContext()
    self.warnings = []
    self._is_module_val = is_module_val
    self.no_mapping = no_mapping
    if has_traits_meth:
      self._get_source_traits = lambda: MockTraits()

  def _get_mapping(self, name, silent=True):
    """Mock implementation of  get mapping."""
    if self.no_mapping:
      return None
    if "requires_plugin" in name:
      return {"requires_plugin": "yes"}
    if "api_found" in name:
      return {"api": name}
    return None

  def _is_stateful(self, name):
    """Mock implementation of  is stateful."""
    return self._is_stateful_val

  def _report_warning(self, msg):
    """Mock implementation of  report warning."""
    self.warnings.append(msg)

  def _get_source_lifecycle_lists(self):
    """Mock implementation of  get source lifecycle lists."""
    return ({"strip_me"}, {"warn_me"})

  def _is_module_alias(self, node):
    """Mock implementation of  is module alias."""
    return self._is_module_val


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=True)
def test_handle_pre_checks_traits_prop(mock_is_functional):
  """Handles pre checks traits prop."""
  rewriter = MockRewriterPre(has_traits_prop=True)
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = cst.Call(
    func=cst.Attribute(value=cst.Name("layer"), attr=cst.Name("apply")),
    args=[cst.Arg(value=cst.Name("vars")), cst.Arg(value=cst.Name("x"))],
  )
  (handled, node) = handle_pre_checks(rewriter, orig, updated, "foo")
  assert handled
  assert isinstance(node.func, cst.Name)
  assert node.func.value == "layer"
  assert len(node.args) == 1
  assert node.args[0].value.value == "x"


@patch("ml_switcheroo.core.rewriter.calls.pre.is_functional_apply", return_value=True)
def test_handle_pre_checks_traits_meth(mock_is_functional):
  """Handles pre checks traits meth."""
  rewriter = MockRewriterPre(has_traits_meth=True)
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = cst.Call(func=cst.Attribute(value=cst.Name("layer"), attr=cst.Name("apply")), args=[])
  (handled, node) = handle_pre_checks(rewriter, orig, updated, "foo")
  assert handled
  assert len(node.args) == 0


def test_handle_pre_checks_plugin_claim():
  """Handles pre checks plugin claim."""
  rewriter = MockRewriterPre(no_mapping=False)
  rewriter.semantics.get_definition.return_value = None
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = orig
  (handled, node) = handle_pre_checks(rewriter, orig, updated, "requires_plugin_func")
  assert not handled
  assert node is updated


def test_handle_pre_checks_is_inplace_and_unroll():
  """Handles pre checks is inplace and unroll."""
  rewriter = MockRewriterPre(no_mapping=True)
  rewriter.semantics.get_definition.return_value = (None, {"is_inplace": True})
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = orig
  with patch("ml_switcheroo.core.rewriter.calls.pre.get_hook") as mock_get_hook:
    mock_hook = MagicMock()
    mock_hook.return_value = cst.Name("unrolled")
    mock_get_hook.return_value = mock_hook
    (handled, node) = handle_pre_checks(rewriter, orig, updated, "foo")
    assert handled
    assert isinstance(node, cst.Name)


def test_handle_pre_checks_endswith_underscore_unroll():
  """Handles pre checks endswith underscore unroll."""
  rewriter = MockRewriterPre(no_mapping=True)
  rewriter.semantics.get_definition.return_value = None
  orig = cst.Call(func=cst.Name("foo_"), args=[])
  updated = orig
  with patch("ml_switcheroo.core.rewriter.calls.pre.get_hook") as mock_get_hook:
    mock_hook = MagicMock()
    mock_hook.return_value = cst.Name("unrolled_")
    mock_get_hook.return_value = mock_hook
    (handled, node) = handle_pre_checks(rewriter, orig, updated, "foo_")
    assert handled
    assert isinstance(node, cst.Name)


def test_handle_pre_checks_lifecycle():
  """Handles pre checks lifecycle."""
  rewriter = MockRewriterPre(no_mapping=True)
  rewriter.semantics.get_definition.return_value = None
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("strip_me")), args=[])
  updated = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("strip_me")), args=[])
  (handled, node) = handle_pre_checks(rewriter, orig, updated, "foo")
  assert handled
  assert isinstance(node, cst.Name)
  assert node.value == "obj"
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("warn_me")), args=[])
  updated = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("warn_me")), args=[])
  (handled, node) = handle_pre_checks(rewriter, orig, updated, "foo")
  assert handled
  assert isinstance(node, cst.Name)
  assert node.value == "obj"


@patch("ml_switcheroo.core.rewriter.calls.pre.rewrite_stateful_call", return_value=cst.Name("stateful_rewritten"))
def test_handle_pre_checks_stateful(mock_rewrite):
  """Handles pre checks stateful."""
  rewriter = MockRewriterPre(is_stateful_val=True, no_mapping=True)
  rewriter.semantics.get_definition.return_value = None
  rewriter.semantics.get_framework_config.return_value = {"stateful_call": {"method": "apply"}}
  orig = cst.Call(func=cst.Name("foo"), args=[])
  updated = orig
  (handled, node) = handle_pre_checks(rewriter, orig, updated, "foo")
  assert handled
  assert isinstance(node, cst.Name)


def test_resolve_implicit_method_self():
  """Resolves implicit method self."""
  rewriter = MockRewriterPre()
  orig = cst.Call(func=cst.Attribute(value=cst.Name("self"), attr=cst.Name("meth")), args=[])
  res = resolve_implicit_method(rewriter, orig, None)
  assert res is None


def test_resolve_implicit_method_module():
  """Resolves implicit method module."""
  rewriter = MockRewriterPre(is_module_val=True)
  orig = cst.Call(func=cst.Attribute(value=cst.Name("mod"), attr=cst.Name("meth")), args=[])
  res = resolve_implicit_method(rewriter, orig, None)
  assert res is None


def test_resolve_implicit_method_sym_table():
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


def test_resolve_implicit_method_legacy_fallback():
  """Resolves implicit method legacy fallback."""
  rewriter = MockRewriterPre(no_mapping=False)
  rewriter.context.symbol_table = None
  rewriter._get_target_traits = MagicMock()
  rewriter.source_traits = MockTraits(implicit_roots=["api_found"])
  orig = cst.Call(func=cst.Attribute(value=cst.Name("obj"), attr=cst.Name("meth")), args=[])
  res = resolve_implicit_method(rewriter, orig, None)
  assert res == "api_found.meth"
  del rewriter.source_traits
  rewriter.semantics.get_framework_config.return_value = {"traits": {"implicit_method_roots": ["api_found"]}}
  res2 = resolve_implicit_method(rewriter, orig, None)
  assert res2 == "api_found.meth"


class MockRule:
  """Mock Rule class for testing purposes."""

  def __init__(self, if_arg, op, is_val=None, use_api=None):
    """Initializes the MockRule instance."""
    self.if_arg = if_arg
    self.op = op
    self.is_val = is_val
    self.use_api = use_api


class MockRewriterDispatch:
  """Mock Rewriter Dispatch class for testing purposes."""

  def __init__(self, source_fw="src", is_module_val=False):
    """Initializes the MockRewriterDispatch instance."""
    self.source_fw = source_fw
    self._is_module_val = is_module_val

  def _is_module_alias(self, node):
    """Mock implementation of  is module alias."""
    return self._is_module_val
