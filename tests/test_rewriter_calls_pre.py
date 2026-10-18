"""Test suite for rewriter calls pre."""

import typing

from typing import Any
from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.core.hooks_registry import clear_hooks, register_hook
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method


class DummyTraits:
  """Test helper class DummyTraits."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.functional_execution_method: typing.Any = "apply"
    self.implicit_method_roots: typing.Any = ["torch.Tensor"]


class DummySemantics:
  """Test helper class DummySemantics."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self._defs: typing.Any = {}
    self._configs: typing.Any = {"torch": {"traits": {}}}

  def get_definition(self, name: typing.Any) -> typing.Any:
    """Execute get definition helper logic.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._defs.get(name)

  def get_framework_config(self, fw: typing.Any) -> typing.Any:
    """Execute get framework config helper logic.

    Args:
        fw: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._configs.get(fw, {})


class DummySymbolTable:
  """Test helper class DummySymbolTable."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.types: typing.Any = {}

  def get_type(self, node: typing.Any) -> Any:
    """Execute get type helper logic.

    Args:
        node: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    if isinstance(node, cst.Name):
      return self.types.get(node.value)
    return None


class DummyContext:
  """Test helper class DummyContext."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.hook_context: typing.Any = {}
    self.symbol_table: typing.Any = DummySymbolTable()


class DummyRewriter:
  """Test helper class DummyRewriter."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.source_fw: typing.Any = "torch"
    self.target_fw: typing.Any = "jax"
    self.source_traits: typing.Any = DummyTraits()
    self.semantics: typing.Any = DummySemantics()
    self.context: typing.Any = DummyContext()
    self._mappings: typing.Any = {}
    self._stateful: typing.Any = []

  def _get_source_traits(self) -> typing.Any:
    """Execute get source traits helper logic.

    Returns:
        Computed test result or fixture value.
    """
    return getattr(self, "source_traits", DummyTraits())

  def _report_warning(self, msg: typing.Any) -> None:
    """Execute report warning helper logic.

    Args:
        msg: Test argument or fixture.
    """
    pass

  def _get_mapping(self, name: typing.Any, silent: typing.Any = False) -> Any:
    """Execute get mapping helper logic.

    Args:
        name: Test argument or fixture.
        silent: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return self._mappings.get(name)

  def _is_stateful(self, name: typing.Any) -> typing.Any:
    """Execute is stateful helper logic.

    Args:
        name: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return name in self._stateful

  def _get_source_lifecycle_lists(self) -> typing.Any:
    """Execute get source lifecycle lists helper logic.

    Returns:
        Computed test result or fixture value.
    """
    return {"strip_me"}, {"warn_me"}

  def _is_module_alias(self, node: typing.Any) -> typing.Any:
    """Execute is module alias helper logic.

    Args:
        node: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return isinstance(node, cst.Name) and node.value == "alias"

  def _get_target_traits(self) -> typing.Any:
    """Execute get target traits helper logic.

    Returns:
        Computed test result or fixture value.
    """
    return DummyTraits()


def test_handle_pre_checks_branches() -> None:
  """Verify handle pre checks branches behavior."""
  rewriter: typing.Any = DummyRewriter()

  node1: typing.Any = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node1, node1, "f") == (False, node1)

  rewriter_no_attr: typing.Any = DummyRewriter()
  del rewriter_no_attr.source_traits
  assert handle_pre_checks(rewriter_no_attr, node1, node1, "f") == (False, node1)

  class NoTraits:
    """Test helper class NoTraits."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.semantics: typing.Any = DummySemantics()

    def _get_target_traits(self) -> typing.Any:
      """Execute get target traits helper logic.

      Returns:
          Computed test result or fixture value.
      """
      return DummyTraits()

  nt_rewriter: typing.Any = NoTraits()
  assert handle_pre_checks(nt_rewriter, node1, node1, "f") == (False, node1)

  node_apply: typing.Any = getattr(getattr(cst.parse_statement("obj.apply(vars, x)"), "body")[0], "value")
  handled: typing.Any
  res: typing.Any
  handled, res = handle_pre_checks(rewriter, node_apply, node_apply, "obj.apply")
  assert handled is True
  assert isinstance(res, cst.Call)

  node_apply_noargs: typing.Any = getattr(getattr(cst.parse_statement("obj.apply()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node_apply_noargs, node_apply_noargs, "obj.apply")[0] is True

  node_apply_orig: typing.Any = getattr(getattr(cst.parse_statement("obj.apply()"), "body")[0], "value")
  node_apply_upd: typing.Any = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node_apply_orig, node_apply_upd, "obj.apply")[0] is False

  # 69 -> 86 func_name is None
  assert handle_pre_checks(rewriter, node1, node1, None) == (False, node1)

  rewriter._mappings["plugin_func"] = {"requires_plugin": True}
  assert handle_pre_checks(rewriter, node1, node1, "plugin_func") == (False, node1)

  class RewriterNoMappingAttr(DummyRewriter):
    """Test helper class RewriterNoMappingAttr."""

    def _get_mapping(self, name: typing.Any, silent: typing.Any = False) -> Any:
      """Execute get mapping helper logic.

      Args:
          name: Test argument or fixture.
          silent: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      raise AttributeError()

  rewriter_no_map_attr: typing.Any = RewriterNoMappingAttr()
  with pytest.raises(AttributeError):
    handle_pre_checks(rewriter_no_map_attr, node1, node1, "plugin_func")

  rewriter._mappings["norm_func"] = {}
  assert handle_pre_checks(rewriter, node1, node1, "norm_func") == (False, node1)

  rewriter.semantics._defs["inplace_func"] = (None, {"is_inplace": True})
  assert handle_pre_checks(rewriter, node1, node1, "inplace_func")[0] is False

  rewriter.semantics._defs["normal_func"] = (None, {})
  assert handle_pre_checks(rewriter, node1, node1, "normal_func")[0] is False

  import ml_switcheroo.core.hooks_registry as hr_module

  clear_hooks()
  hr_module._PLUGINS_LOADED = True

  def dummy_hook(node: typing.Any, ctx: Any) -> typing.Any:
    """Execute dummy hook helper logic.

    Args:
        node: Test argument or fixture.
        ctx: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return cst.Name("changed")

  register_hook("unroll_inplace_ops")(dummy_hook)
  assert handle_pre_checks(rewriter, node1, node1, "inplace_func")[0] is True

  def dummy_hook_same(node: typing.Any, ctx: Any) -> typing.Any:
    """Execute dummy hook same helper logic.

    Args:
        node: Test argument or fixture.
        ctx: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return node

  clear_hooks()
  hr_module._PLUGINS_LOADED = True
  register_hook("unroll_inplace_ops")(dummy_hook_same)
  assert handle_pre_checks(rewriter, node1, node1, "inplace_func")[0] is False

  clear_hooks()
  hr_module._PLUGINS_LOADED = True
  assert handle_pre_checks(rewriter, node1, node1, "inplace_func")[0] is False

  assert handle_pre_checks(rewriter, node1, node1, "add_")[0] is False

  assert handle_pre_checks(rewriter, node1, node1, "add")[0] is False

  node_attr: typing.Any = getattr(getattr(cst.parse_statement("obj.strip_me()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node_attr, node_attr, "obj.strip_me")[0] is True

  node_attr_upd: typing.Any = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node_attr, node_attr_upd, "obj.strip_me")[0] is False

  node_warn: typing.Any = getattr(getattr(cst.parse_statement("obj.warn_me()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node_warn, node_warn, "obj.warn_me")[0] is True

  assert handle_pre_checks(rewriter, node_warn, node_attr_upd, "obj.warn_me")[0] is False

  assert handle_pre_checks(rewriter, node1, node1, "f")[0] is False

  class RewriterNoLifecycle(DummyRewriter):
    """Test helper class RewriterNoLifecycle."""

    def _get_source_lifecycle_lists(self) -> typing.Any:
      """Execute get source lifecycle lists helper logic.

      Returns:
          Computed test result or fixture value.
      """
      raise AttributeError()

  rewriter_no_lifecycle: typing.Any = RewriterNoLifecycle()
  with pytest.raises(AttributeError):
    handle_pre_checks(rewriter_no_lifecycle, node_attr, node_attr, "obj.strip_me")

  rewriter._stateful.append("state_f")
  rewriter.semantics._configs["jax"] = {"stateful_call": {"method": "mock", "prepend_arg": "vars"}}
  assert handle_pre_checks(rewriter, node1, node1, "state_f")[0] is True

  rewriter.semantics._configs["jax"] = {}
  assert handle_pre_checks(rewriter, node1, node1, "state_f")[0] is False


def test_resolve_implicit_method_branches() -> None:
  """Verify resolve implicit method branches behavior."""
  rewriter: typing.Any = DummyRewriter()

  node_attr: typing.Any = getattr(getattr(cst.parse_statement("obj.f()"), "body")[0], "value")
  assert resolve_implicit_method(rewriter, node_attr, "obj.f") is None

  node_name: typing.Any = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  assert resolve_implicit_method(rewriter, node_name, "f") is None

  node_self: typing.Any = getattr(getattr(cst.parse_statement("self.f()"), "body")[0], "value")
  assert resolve_implicit_method(rewriter, node_self, "self.f") is None

  node_mod: typing.Any = getattr(getattr(cst.parse_statement("alias.f()"), "body")[0], "value")
  assert resolve_implicit_method(rewriter, node_mod, "alias.f") is None

  class RewriterNoModuleAttr(DummyRewriter):
    """Test helper class RewriterNoModuleAttr."""

    def _is_module_alias(self, node: typing.Any) -> typing.Any:
      """Execute is module alias helper logic.

      Args:
          node: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      raise AttributeError()

  rewriter_no_attr_alias: typing.Any = RewriterNoModuleAttr()
  with pytest.raises(AttributeError):
    resolve_implicit_method(rewriter_no_attr_alias, node_attr, "obj.f")

  # Completely missing method
  class RewriterMissingModuleAttr(DummyRewriter):
    """Test helper class RewriterMissingModuleAttr."""

    def _is_module_alias(self, node: typing.Any) -> typing.Any:
      """Execute is module alias helper logic.

      Args:
          node: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      raise AttributeError()

  rm_attr: typing.Any = RewriterMissingModuleAttr()
  with pytest.raises(AttributeError):
    resolve_implicit_method(rm_attr, node_attr, "obj.f")

  class DummyType:
    """Test helper class DummyType."""

    def __init__(self, n: typing.Any, fw: typing.Any = None) -> None:
      """Initialize test helper instance.

      Args:
          n: Test argument or fixture.
          fw: Test argument or fixture.
      """
      self.name = n
      if fw:
        self.framework = fw

  class RewriterNoAlias(DummyRewriter):
    """Test helper class RewriterNoAlias."""

    def _is_module_alias(self, node: typing.Any) -> typing.Any:
      """Execute is module alias helper logic.

      Args:
          node: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return False

  rewriter_no_alias: typing.Any = RewriterNoAlias()
  rewriter_no_alias.context.symbol_table.types["obj"] = DummyType("MyType")
  rewriter_no_alias._mappings["MyType.f"] = {"test": True}
  assert resolve_implicit_method(rewriter_no_alias, node_attr, "obj.f") == "MyType.f"

  # 159 -> 174 context missing symbol_table
  class ContextNoSym:
    """Test helper class ContextNoSym."""

    pass

  rewriter_no_alias.context = ContextNoSym()
  with pytest.raises(AttributeError):
    resolve_implicit_method(rewriter_no_alias, node_attr, "obj.f")
  rewriter_no_alias.context = DummyContext()

  rewriter_no_alias.context.symbol_table.types["obj2"] = DummyType("Tensor")  # no framework
  assert (
    resolve_implicit_method(
      rewriter_no_alias, getattr(getattr(cst.parse_statement("obj2.f()"), "body")[0], "value"), "obj2.f"
    )
    is None
  )

  # Ensure mapping returns False to cover 170 -> 174
  rewriter_no_alias._mappings["MyType.g"] = None
  assert (
    resolve_implicit_method(
      rewriter_no_alias, getattr(getattr(cst.parse_statement("obj.g()"), "body")[0], "value"), "obj.g"
    )
    is None
  )

  rewriter_no_alias.context.symbol_table.types["t"] = DummyType("Tensor", fw="torch")
  node_tensor: typing.Any = getattr(getattr(cst.parse_statement("t.f()"), "body")[0], "value")
  rewriter_no_alias._mappings["torch.Tensor.f"] = {"test": True}
  assert resolve_implicit_method(rewriter_no_alias, node_tensor, "t.f") == "torch.Tensor.f"

  class RewriterNoMapping(DummyRewriter):
    """Test helper class RewriterNoMapping."""

    def _get_mapping(self, name: typing.Any, silent: typing.Any = False) -> Any:
      """Execute get mapping helper logic.

      Args:
          name: Test argument or fixture.
          silent: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      raise AttributeError()

  rnm: typing.Any = RewriterNoMapping()
  rnm.context.symbol_table.types["obj"] = DummyType("MyType")
  with pytest.raises(AttributeError):
    resolve_implicit_method(rnm, node_attr, "obj.f")

  # No get_mapping on rewriter at all
  class RewriterMissingMappingAttr(DummyRewriter):
    """Test helper class RewriterMissingMappingAttr."""

    def _get_mapping(self, name: typing.Any, silent: typing.Any = False) -> Any:
      """Execute get mapping helper logic.

      Args:
          name: Test argument or fixture.
          silent: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      raise AttributeError()

  r_no_map: typing.Any = RewriterMissingMappingAttr()
  r_no_map.context.symbol_table.types["obj"] = DummyType("MyType")
  with pytest.raises(AttributeError):
    resolve_implicit_method(r_no_map, node_attr, "obj.f")

  node_unknown: typing.Any = getattr(getattr(cst.parse_statement("unknown.f()"), "body")[0], "value")

  class DummyTargetRewriter(DummyRewriter):
    """Test helper class DummyTargetRewriter."""

    pass

  tr: typing.Any = DummyTargetRewriter()
  tr._mappings["torch.Tensor.f"] = {"test": True}
  assert resolve_implicit_method(tr, node_unknown, "unknown.f") == "torch.Tensor.f"

  # Not found in _mappings
  assert (
    resolve_implicit_method(tr, getattr(getattr(cst.parse_statement("unknown.g()"), "body")[0], "value"), "unknown.g")
    is None
  )

  class TargetNoMap(DummyTargetRewriter):
    """Test helper class TargetNoMap."""

    def _get_mapping(self, name: typing.Any, silent: typing.Any = False) -> Any:
      """Execute get mapping helper logic.

      Args:
          name: Test argument or fixture.
          silent: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      raise AttributeError()

  tr_no_map: typing.Any = TargetNoMap()
  with pytest.raises(AttributeError):
    resolve_implicit_method(tr_no_map, node_unknown, "unknown.f")

  class TargetMissingMapAttr(DummyTargetRewriter):
    """Test helper class TargetMissingMapAttr."""

    def _get_mapping(self, name: typing.Any, silent: typing.Any = False) -> Any:
      """Execute get mapping helper logic.

      Args:
          name: Test argument or fixture.
          silent: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      raise AttributeError()

  tr_missing_map: typing.Any = TargetMissingMapAttr()
  with pytest.raises(AttributeError):
    resolve_implicit_method(tr_missing_map, node_unknown, "unknown.f")

  tr_no_traits: typing.Any = DummyTargetRewriter()
  del tr_no_traits.source_traits
  tr_no_traits.semantics._configs["torch"] = {"traits": {"implicit_method_roots": ["torch.Tensor"]}}
  assert resolve_implicit_method(tr_no_traits, node_unknown, "unknown.f") is None

  # 174 -> 194 (no _get_target_traits)
  class TargetNoTraitsAttr(DummyTargetRewriter):
    """Test helper class TargetNoTraitsAttr."""

    def _get_target_traits(self) -> typing.Any:
      """Execute get target traits helper logic.

      Returns:
          Computed test result or fixture value.
      """
      raise AttributeError()

  tnt: typing.Any = TargetNoTraitsAttr()
  assert resolve_implicit_method(tnt, node_unknown, "unknown.f") is None


def test_handle_pre_checks_no_hook() -> None:
  """Verify handle pre checks no hook behavior."""
  from unittest.mock import MagicMock, patch

  import libcst as cst

  from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks

  class DummyRewriter:
    """Test helper class DummyRewriter."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.semantics: typing.Any = MagicMock()
      self.semantics.get_definition.return_value = None

  rewriter: typing.Any = DummyRewriter()
  original: typing.Any = getattr(getattr(cst.parse_statement("f_()"), "body")[0], "value")

  with patch("ml_switcheroo.core.rewriter.calls.pre.get_hook", return_value=None):
    res: typing.Any
    updated: typing.Any
    res, updated = handle_pre_checks(rewriter, original, original, "f_")
    assert not res


def test_resolve_implicit_method_missing_attributes() -> None:
  """Verify resolve implicit method missing attributes behavior."""
  from unittest.mock import MagicMock

  import libcst as cst

  from ml_switcheroo.core.rewriter.calls.pre import resolve_implicit_method

  # Branch [154, 157]: Missing _is_module_alias
  class RewriterNoIsModule:
    """Test helper class RewriterNoIsModule."""

    pass

  rewriter: typing.Any = RewriterNoIsModule()
  node_attr: typing.Any = getattr(getattr(cst.parse_statement("obj.f()"), "body")[0], "value")
  assert resolve_implicit_method(rewriter, node_attr, "obj.f") is None

  # Branch [168, 174]: Has symbol table, but missing _get_mapping
  class SymType:
    """Test helper class SymType."""

    name: typing.Any = "Tensor"
    framework: typing.Any = "torch"

  class SymTable:
    """Test helper class SymTable."""

    def get_type(self, node: typing.Any) -> typing.Any:
      """Execute get type helper logic.

      Args:
          node: Test argument or fixture.

      Returns:
          Computed test result or fixture value.
      """
      return SymType()

  class RewriterSymTableNoMap:
    """Test helper class RewriterSymTableNoMap."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.context: typing.Any = MagicMock()
      self.context.symbol_table = SymTable()

  r2: typing.Any = RewriterSymTableNoMap()
  assert resolve_implicit_method(r2, node_attr, "obj.f") is None

  # Branch [189, 187]: Fallback legacy, missing _get_mapping
  class RewriterLegacyNoMap:
    """Test helper class RewriterLegacyNoMap."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.source_traits: typing.Any = MagicMock()
      self.source_traits.implicit_method_roots = ["torch.Tensor", "numpy.ndarray"]

    def _get_target_traits(self) -> None:
      """Execute get target traits helper logic."""
      pass

  r3: typing.Any = RewriterLegacyNoMap()
  assert resolve_implicit_method(r3, node_attr, "obj.f") is None


# --- Merged from test_rewriter_calls_pre_extra.py ---


class DummyRewriterExtra:
  """Test helper class DummyRewriterExtra."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.context: typing.Any = MagicMock()
    self.context.hook_context = MagicMock()
    self.semantics: typing.Any = MagicMock()
    self.semantics.get_definition.return_value = None

  def _report_warning(self, w: typing.Any) -> None:
    """Execute report warning helper logic.

    Args:
        w: Test argument or fixture.
    """
    pass


@pytest.fixture(autouse=True)
def _cleanup() -> typing.Any:
  """Execute cleanup helper logic.

  Returns:
      Computed test result or fixture value.
  """
  import ml_switcheroo.core.hooks_registry as hr

  hr.clear_hooks()
  hr._PLUGINS_LOADED = True
  yield
  hr.clear_hooks()


def test_handle_pre_checks_inplace_no_change() -> None:
  """Verify handle pre checks inplace no change behavior."""
  rewriter: typing.Any = DummyRewriter()

  # 97-98: in-place unroll hook doesn't change node
  @register_hook("unroll_inplace_ops")
  def mock_hook(node: typing.Any, ctx: typing.Any) -> typing.Any:
    """Execute mock hook helper logic.

    Args:
        node: Test argument or fixture.
        ctx: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return node

  original: typing.Any = getattr(getattr(cst.parse_statement("foo_()"), "body")[0], "value")
  res: typing.Any = handle_pre_checks(rewriter, original, original, "foo_")
  assert res[0] is False


def test_handle_pre_checks_inplace_change() -> None:
  """Verify handle pre checks inplace change behavior."""
  rewriter: typing.Any = DummyRewriter()

  # 97-98: in-place unroll hook changes node
  @register_hook("unroll_inplace_ops")
  def mock_hook(node: typing.Any, ctx: typing.Any) -> typing.Any:
    """Execute mock hook helper logic.

    Args:
        node: Test argument or fixture.
        ctx: Test argument or fixture.

    Returns:
        Computed test result or fixture value.
    """
    return getattr(cst.parse_statement("b = 1"), "body")[0]

  original: typing.Any = getattr(getattr(cst.parse_statement("foo_()"), "body")[0], "value")
  res: typing.Any = handle_pre_checks(rewriter, original, original, "foo_")
  assert res[0] is True
