"""Test suite for rewriter calls pre."""

from typing import Any, Dict, Generator, List, Optional, Set, Tuple
from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.core.hooks_registry import clear_hooks, register_hook
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method


class DummyTraits:
  """Test helper class DummyTraits."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.functional_execution_method: str = "apply"
    self.implicit_method_roots: List[str] = ["torch.Tensor"]


class DummySemantics:
  """Test helper class DummySemantics."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self._defs: Dict[str, Tuple[Optional[str], Dict[str, Any]]] = {}
    self._configs: Dict[str, Dict[str, Any]] = {"torch": {"traits": {}}}

  def get_definition(self, name: str) -> Optional[Tuple[Optional[str], Dict[str, Any]]]:
    """Execute get definition helper logic."""
    return self._defs.get(name)

  def get_framework_config(self, fw: str) -> Dict[str, Any]:
    """Execute get framework config helper logic."""
    return self._configs.get(fw, {})


class DummySymbolTable:
  """Test helper class DummySymbolTable."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.types: Dict[str, Any] = {}

  def get_type(self, node: cst.CSTNode) -> Any:
    """Execute get type helper logic."""
    if isinstance(node, cst.Name):
      return self.types.get(node.value)
    return None


class DummyContext:
  """Test helper class DummyContext."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.hook_context: Dict[str, Any] = {}
    self.symbol_table: DummySymbolTable = DummySymbolTable()


class DummyRewriter:
  """Test helper class DummyRewriter."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.source_fw: str = "torch"
    self.target_fw: str = "jax"
    self.source_traits: DummyTraits = DummyTraits()
    self.semantics: DummySemantics = DummySemantics()
    self.context: DummyContext = DummyContext()
    self._mappings: Dict[str, Any] = {}
    self._stateful: List[str] = []

  def _get_source_traits(self) -> DummyTraits:
    """Execute get source traits helper logic."""
    return getattr(self, "source_traits", DummyTraits())

  def _report_warning(self, msg: str) -> None:
    """Execute report warning helper logic."""
    pass

  def _get_mapping(self, name: str, silent: bool = False) -> Any:
    """Execute get mapping helper logic."""
    return self._mappings.get(name)

  def _is_stateful(self, name: str) -> bool:
    """Execute is stateful helper logic."""
    return name in self._stateful

  def _get_source_lifecycle_lists(self) -> Tuple[Set[str], Set[str]]:
    """Execute get source lifecycle lists helper logic."""
    return {"strip_me"}, {"warn_me"}

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """Execute is module alias helper logic."""
    return isinstance(node, cst.Name) and node.value == "alias"

  def _get_target_traits(self) -> DummyTraits:
    """Execute get target traits helper logic."""
    return DummyTraits()


def test_handle_pre_checks_branches() -> None:
  """Verify handle pre checks branches behavior."""
  rewriter: DummyRewriter = DummyRewriter()

  node1: cst.Call = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node1, node1, "f") == (False, node1)

  rewriter_no_attr: DummyRewriter = DummyRewriter()
  del rewriter_no_attr.source_traits
  assert handle_pre_checks(rewriter_no_attr, node1, node1, "f") == (False, node1)

  class NoTraits:
    """Test helper class NoTraits."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.semantics: DummySemantics = DummySemantics()

    def _get_target_traits(self) -> DummyTraits:
      """Execute get target traits helper logic."""
      return DummyTraits()

  nt_rewriter: NoTraits = NoTraits()
  assert handle_pre_checks(nt_rewriter, node1, node1, "f") == (False, node1)

  node_apply: cst.Call = getattr(getattr(cst.parse_statement("obj.apply(vars, x)"), "body")[0], "value")
  handled: bool
  res: cst.CSTNode
  handled, res = handle_pre_checks(rewriter, node_apply, node_apply, "obj.apply")
  assert handled is True
  assert isinstance(res, cst.Call)

  node_apply_noargs: cst.Call = getattr(getattr(cst.parse_statement("obj.apply()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node_apply_noargs, node_apply_noargs, "obj.apply")[0] is True

  node_apply_orig: cst.Call = getattr(getattr(cst.parse_statement("obj.apply()"), "body")[0], "value")
  node_apply_upd: cst.Call = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node_apply_orig, node_apply_upd, "obj.apply")[0] is False

  # 69 -> 86 func_name is None
  assert handle_pre_checks(rewriter, node1, node1, None) == (False, node1)

  rewriter._mappings["plugin_func"] = {"requires_plugin": True}
  assert handle_pre_checks(rewriter, node1, node1, "plugin_func") == (False, node1)

  class RewriterNoMappingAttr(DummyRewriter):
    """Test helper class RewriterNoMappingAttr."""

    def _get_mapping(self, name: str, silent: bool = False) -> Any:
      """Execute get mapping helper logic."""
      raise AttributeError()

  rewriter_no_map_attr: RewriterNoMappingAttr = RewriterNoMappingAttr()
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

  def dummy_hook(node: cst.CSTNode, ctx: Any) -> cst.CSTNode:
    """Execute dummy hook helper logic."""
    return cst.Name("changed")

  register_hook("unroll_inplace_ops")(dummy_hook)
  assert handle_pre_checks(rewriter, node1, node1, "inplace_func")[0] is True

  def dummy_hook_same(node: cst.CSTNode, ctx: Any) -> cst.CSTNode:
    """Execute dummy hook same helper logic."""
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

  node_attr: cst.Call = getattr(getattr(cst.parse_statement("obj.strip_me()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node_attr, node_attr, "obj.strip_me")[0] is True

  node_attr_upd: cst.Call = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node_attr, node_attr_upd, "obj.strip_me")[0] is False

  node_warn: cst.Call = getattr(getattr(cst.parse_statement("obj.warn_me()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node_warn, node_warn, "obj.warn_me")[0] is True

  assert handle_pre_checks(rewriter, node_warn, node_attr_upd, "obj.warn_me")[0] is False

  assert handle_pre_checks(rewriter, node1, node1, "f")[0] is False

  class RewriterNoLifecycle(DummyRewriter):
    """Test helper class RewriterNoLifecycle."""

    def _get_source_lifecycle_lists(self) -> Tuple[Set[str], Set[str]]:
      """Execute get source lifecycle lists helper logic."""
      raise AttributeError()

  rewriter_no_lifecycle: RewriterNoLifecycle = RewriterNoLifecycle()
  with pytest.raises(AttributeError):
    handle_pre_checks(rewriter_no_lifecycle, node_attr, node_attr, "obj.strip_me")

  rewriter._stateful.append("state_f")
  rewriter.semantics._configs["jax"] = {"stateful_call": {"method": "mock", "prepend_arg": "vars"}}
  assert handle_pre_checks(rewriter, node1, node1, "state_f")[0] is True

  rewriter.semantics._configs["jax"] = {}
  assert handle_pre_checks(rewriter, node1, node1, "state_f")[0] is False


def test_resolve_implicit_method_branches() -> None:
  """Verify resolve implicit method branches behavior."""
  rewriter: DummyRewriter = DummyRewriter()

  node_attr: cst.Call = getattr(getattr(cst.parse_statement("obj.f()"), "body")[0], "value")
  assert resolve_implicit_method(rewriter, node_attr, "obj.f") is None

  node_name: cst.Call = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  assert resolve_implicit_method(rewriter, node_name, "f") is None

  node_self: cst.Call = getattr(getattr(cst.parse_statement("self.f()"), "body")[0], "value")
  assert resolve_implicit_method(rewriter, node_self, "self.f") is None

  node_mod: cst.Call = getattr(getattr(cst.parse_statement("alias.f()"), "body")[0], "value")
  assert resolve_implicit_method(rewriter, node_mod, "alias.f") is None

  class RewriterNoModuleAttr(DummyRewriter):
    """Test helper class RewriterNoModuleAttr."""

    def _is_module_alias(self, node: cst.CSTNode) -> bool:
      """Execute is module alias helper logic."""
      raise AttributeError()

  rewriter_no_attr_alias: RewriterNoModuleAttr = RewriterNoModuleAttr()
  with pytest.raises(AttributeError):
    resolve_implicit_method(rewriter_no_attr_alias, node_attr, "obj.f")

  # Completely missing method
  class RewriterMissingModuleAttr(DummyRewriter):
    """Test helper class RewriterMissingModuleAttr."""

    def _is_module_alias(self, node: cst.CSTNode) -> bool:
      """Execute is module alias helper logic."""
      raise AttributeError()

  rm_attr: RewriterMissingModuleAttr = RewriterMissingModuleAttr()
  with pytest.raises(AttributeError):
    resolve_implicit_method(rm_attr, node_attr, "obj.f")

  class DummyType:
    """Test helper class DummyType."""

    def __init__(self, n: str, fw: Optional[str] = None) -> None:
      """Initialize test helper instance."""
      self.name = n
      if fw:
        self.framework = fw

  class RewriterNoAlias(DummyRewriter):
    """Test helper class RewriterNoAlias."""

    def _is_module_alias(self, node: cst.CSTNode) -> bool:
      """Execute is module alias helper logic."""
      return False

  rewriter_no_alias: RewriterNoAlias = RewriterNoAlias()
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
  node_tensor: cst.Call = getattr(getattr(cst.parse_statement("t.f()"), "body")[0], "value")
  rewriter_no_alias._mappings["torch.Tensor.f"] = {"test": True}
  assert resolve_implicit_method(rewriter_no_alias, node_tensor, "t.f") == "torch.Tensor.f"

  class RewriterNoMapping(DummyRewriter):
    """Test helper class RewriterNoMapping."""

    def _get_mapping(self, name: str, silent: bool = False) -> Any:
      """Execute get mapping helper logic."""
      raise AttributeError()

  rnm: RewriterNoMapping = RewriterNoMapping()
  rnm.context.symbol_table.types["obj"] = DummyType("MyType")
  with pytest.raises(AttributeError):
    resolve_implicit_method(rnm, node_attr, "obj.f")

  # No get_mapping on rewriter at all
  class RewriterMissingMappingAttr(DummyRewriter):
    """Test helper class RewriterMissingMappingAttr."""

    def _get_mapping(self, name: str, silent: bool = False) -> Any:
      """Execute get mapping helper logic."""
      raise AttributeError()

  r_no_map: RewriterMissingMappingAttr = RewriterMissingMappingAttr()
  r_no_map.context.symbol_table.types["obj"] = DummyType("MyType")
  with pytest.raises(AttributeError):
    resolve_implicit_method(r_no_map, node_attr, "obj.f")

  node_unknown: cst.Call = getattr(getattr(cst.parse_statement("unknown.f()"), "body")[0], "value")

  class DummyTargetRewriter(DummyRewriter):
    """Test helper class DummyTargetRewriter."""

    pass

  tr: DummyTargetRewriter = DummyTargetRewriter()
  tr._mappings["torch.Tensor.f"] = {"test": True}
  assert resolve_implicit_method(tr, node_unknown, "unknown.f") == "torch.Tensor.f"

  # Not found in _mappings
  assert (
    resolve_implicit_method(tr, getattr(getattr(cst.parse_statement("unknown.g()"), "body")[0], "value"), "unknown.g")
    is None
  )

  class TargetNoMap(DummyTargetRewriter):
    """Test helper class TargetNoMap."""

    def _get_mapping(self, name: str, silent: bool = False) -> Any:
      """Execute get mapping helper logic."""
      raise AttributeError()

  tr_no_map: TargetNoMap = TargetNoMap()
  with pytest.raises(AttributeError):
    resolve_implicit_method(tr_no_map, node_unknown, "unknown.f")

  class TargetMissingMapAttr(DummyTargetRewriter):
    """Test helper class TargetMissingMapAttr."""

    def _get_mapping(self, name: str, silent: bool = False) -> Any:
      """Execute get mapping helper logic."""
      raise AttributeError()

  tr_missing_map: TargetMissingMapAttr = TargetMissingMapAttr()
  with pytest.raises(AttributeError):
    resolve_implicit_method(tr_missing_map, node_unknown, "unknown.f")

  tr_no_traits: DummyTargetRewriter = DummyTargetRewriter()
  del tr_no_traits.source_traits
  tr_no_traits.semantics._configs["torch"] = {"traits": {"implicit_method_roots": ["torch.Tensor"]}}
  assert resolve_implicit_method(tr_no_traits, node_unknown, "unknown.f") is None

  # 174 -> 194 (no _get_target_traits)
  class TargetNoTraitsAttr(DummyTargetRewriter):
    """Test helper class TargetNoTraitsAttr."""

    def _get_target_traits(self) -> DummyTraits:
      """Execute get target traits helper logic."""
      raise AttributeError()

  tnt: TargetNoTraitsAttr = TargetNoTraitsAttr()
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
      self.semantics: MagicMock = MagicMock()
      self.semantics.get_definition.return_value = None

  rewriter: DummyRewriter = DummyRewriter()
  original: cst.Call = getattr(getattr(cst.parse_statement("f_()"), "body")[0], "value")

  with patch("ml_switcheroo.core.rewriter.calls.pre.get_hook", return_value=None):
    res: bool
    updated: cst.CSTNode
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

  rewriter: RewriterNoIsModule = RewriterNoIsModule()
  node_attr: cst.Call = getattr(getattr(cst.parse_statement("obj.f()"), "body")[0], "value")
  assert resolve_implicit_method(rewriter, node_attr, "obj.f") is None

  # Branch [168, 174]: Has symbol table, but missing _get_mapping
  class SymType:
    """Test helper class SymType."""

    name: str = "Tensor"
    framework: str = "torch"

  class SymTable:
    """Test helper class SymTable."""

    def get_type(self, node: cst.CSTNode) -> SymType:
      """Execute get type helper logic."""
      return SymType()

  class RewriterSymTableNoMap:
    """Test helper class RewriterSymTableNoMap."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.context: MagicMock = MagicMock()
      self.context.symbol_table = SymTable()

  r2: RewriterSymTableNoMap = RewriterSymTableNoMap()
  assert resolve_implicit_method(r2, node_attr, "obj.f") is None

  # Branch [189, 187]: Fallback legacy, missing _get_mapping
  class RewriterLegacyNoMap:
    """Test helper class RewriterLegacyNoMap."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.source_traits: MagicMock = MagicMock()
      self.source_traits.implicit_method_roots = ["torch.Tensor", "numpy.ndarray"]

    def _get_target_traits(self) -> None:
      """Execute get target traits helper logic."""
      pass

  r3: RewriterLegacyNoMap = RewriterLegacyNoMap()
  assert resolve_implicit_method(r3, node_attr, "obj.f") is None


# --- Merged from test_rewriter_calls_pre_extra.py ---


class DummyRewriterExtra:
  """Test helper class DummyRewriterExtra."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.context: MagicMock = MagicMock()
    self.context.hook_context = MagicMock()
    self.semantics: MagicMock = MagicMock()
    self.semantics.get_definition.return_value = None

  def _report_warning(self, w: str) -> None:
    """Execute report warning helper logic."""
    pass


@pytest.fixture(autouse=True)
def _cleanup() -> Generator[None, None, None]:
  """Execute cleanup helper logic."""
  import ml_switcheroo.core.hooks_registry as hr

  hr.clear_hooks()
  hr._PLUGINS_LOADED = True
  yield
  hr.clear_hooks()


def test_handle_pre_checks_inplace_no_change() -> None:
  """Verify handle pre checks inplace no change behavior."""
  rewriter: DummyRewriter = DummyRewriter()

  # 97-98: in-place unroll hook doesn't change node
  @register_hook("unroll_inplace_ops")
  def mock_hook(node: cst.CSTNode, ctx: MagicMock) -> cst.CSTNode:
    """Execute mock hook helper logic."""
    return node

  original: cst.Call = getattr(getattr(cst.parse_statement("foo_()"), "body")[0], "value")
  res: Tuple[bool, cst.CSTNode] = handle_pre_checks(rewriter, original, original, "foo_")
  assert res[0] is False


def test_handle_pre_checks_inplace_change() -> None:
  """Verify handle pre checks inplace change behavior."""
  rewriter: DummyRewriter = DummyRewriter()

  # 97-98: in-place unroll hook changes node
  @register_hook("unroll_inplace_ops")
  def mock_hook(node: cst.CSTNode, ctx: MagicMock) -> cst.CSTNode:
    """Execute mock hook helper logic."""
    return getattr(cst.parse_statement("b = 1"), "body")[0]

  original: cst.Call = getattr(getattr(cst.parse_statement("foo_()"), "body")[0], "value")
  res: Tuple[bool, cst.CSTNode] = handle_pre_checks(rewriter, original, original, "foo_")
  assert res[0] is True
