"""Module docstring."""

from typing import Any, Dict, Generator, List, Optional, Set, Tuple
from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.core.hooks_registry import clear_hooks, register_hook
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method


class DummyTraits:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.functional_execution_method: str = "apply"
    self.implicit_method_roots: List[str] = ["torch.Tensor"]


class DummySemantics:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self._defs: Dict[str, Tuple[Optional[str], Dict[str, Any]]] = {}
    self._configs: Dict[str, Dict[str, Any]] = {"torch": {"traits": {}}}

  def get_definition(self, name: str) -> Optional[Tuple[Optional[str], Dict[str, Any]]]:
    """Docstring."""
    return self._defs.get(name)

  def get_framework_config(self, fw: str) -> Dict[str, Any]:
    """Docstring."""
    return self._configs.get(fw, {})


class DummySymbolTable:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.types: Dict[str, Any] = {}

  def get_type(self, node: cst.CSTNode) -> Any:
    """Docstring."""
    if isinstance(node, cst.Name):
      return self.types.get(node.value)
    return None


class DummyContext:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.hook_context: Dict[str, Any] = {}
    self.symbol_table: DummySymbolTable = DummySymbolTable()


class DummyRewriter:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.source_fw: str = "torch"
    self.target_fw: str = "jax"
    self.source_traits: DummyTraits = DummyTraits()
    self.semantics: DummySemantics = DummySemantics()
    self.context: DummyContext = DummyContext()
    self._mappings: Dict[str, Any] = {}
    self._stateful: List[str] = []

  def _get_source_traits(self) -> DummyTraits:
    """Function doc."""
    return getattr(self, "source_traits", DummyTraits())

  def _report_warning(self, msg: str) -> None:
    """Function doc."""
    pass

  def _get_mapping(self, name: str, silent: bool = False) -> Any:
    """Function doc."""
    return self._mappings.get(name)

  def _is_stateful(self, name: str) -> bool:
    """Function doc."""
    return name in self._stateful

  def _get_source_lifecycle_lists(self) -> Tuple[Set[str], Set[str]]:
    """Function doc."""
    return {"strip_me"}, {"warn_me"}

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """Function doc."""
    return isinstance(node, cst.Name) and node.value == "alias"

  def _get_target_traits(self) -> DummyTraits:
    """Function doc."""
    return DummyTraits()


def test_handle_pre_checks_branches() -> None:
  """Docstring."""
  rewriter: DummyRewriter = DummyRewriter()

  node1: cst.Call = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  handle_pre_checks(rewriter, node1, node1, "f")

  rewriter_no_attr: DummyRewriter = DummyRewriter()
  del rewriter_no_attr.source_traits
  handle_pre_checks(rewriter_no_attr, node1, node1, "f")

  class NoTraits:
    """Class doc."""

    def __init__(self) -> None:
      """Init doc."""
      self.semantics: DummySemantics = DummySemantics()

    def _get_target_traits(self) -> DummyTraits:
      """Function doc."""
      return DummyTraits()

  nt_rewriter: NoTraits = NoTraits()
  handle_pre_checks(nt_rewriter, node1, node1, "f")

  node_apply: cst.Call = getattr(getattr(cst.parse_statement("obj.apply(vars, x)"), "body")[0], "value")
  handled: bool
  res: cst.CSTNode
  handled, res = handle_pre_checks(rewriter, node_apply, node_apply, "obj.apply")
  assert handled

  node_apply_noargs: cst.Call = getattr(getattr(cst.parse_statement("obj.apply()"), "body")[0], "value")
  handle_pre_checks(rewriter, node_apply_noargs, node_apply_noargs, "obj.apply")

  node_apply_orig: cst.Call = getattr(getattr(cst.parse_statement("obj.apply()"), "body")[0], "value")
  node_apply_upd: cst.Call = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  handle_pre_checks(rewriter, node_apply_orig, node_apply_upd, "obj.apply")

  # 69 -> 86 func_name is None
  handle_pre_checks(rewriter, node1, node1, None)

  rewriter._mappings["plugin_func"] = {"requires_plugin": True}
  handle_pre_checks(rewriter, node1, node1, "plugin_func")

  class RewriterNoMappingAttr(DummyRewriter):
    """Class doc."""

    def _get_mapping(self, name: str, silent: bool = False) -> Any:
      """Function doc."""
      raise AttributeError()

  rewriter_no_map_attr: RewriterNoMappingAttr = RewriterNoMappingAttr()
  try:
    handle_pre_checks(rewriter_no_map_attr, node1, node1, "plugin_func")
  except AttributeError:
    pass

  rewriter._mappings["norm_func"] = {}
  handle_pre_checks(rewriter, node1, node1, "norm_func")

  rewriter.semantics._defs["inplace_func"] = (None, {"is_inplace": True})
  handle_pre_checks(rewriter, node1, node1, "inplace_func")

  rewriter.semantics._defs["normal_func"] = (None, {})
  handle_pre_checks(rewriter, node1, node1, "normal_func")

  clear_hooks()

  def dummy_hook(node: cst.CSTNode, ctx: Any) -> cst.CSTNode:
    """Function doc."""
    return cst.Name("changed")

  register_hook("unroll_inplace_ops")(dummy_hook)
  handle_pre_checks(rewriter, node1, node1, "inplace_func")

  def dummy_hook_same(node: cst.CSTNode, ctx: Any) -> cst.CSTNode:
    """Function doc."""
    return node

  clear_hooks()
  register_hook("unroll_inplace_ops")(dummy_hook_same)
  handle_pre_checks(rewriter, node1, node1, "inplace_func")

  clear_hooks()
  handle_pre_checks(rewriter, node1, node1, "inplace_func")

  handle_pre_checks(rewriter, node1, node1, "add_")

  handle_pre_checks(rewriter, node1, node1, "add")

  node_attr: cst.Call = getattr(getattr(cst.parse_statement("obj.strip_me()"), "body")[0], "value")
  handle_pre_checks(rewriter, node_attr, node_attr, "obj.strip_me")

  node_attr_upd: cst.Call = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  handle_pre_checks(rewriter, node_attr, node_attr_upd, "obj.strip_me")

  node_warn: cst.Call = getattr(getattr(cst.parse_statement("obj.warn_me()"), "body")[0], "value")
  handle_pre_checks(rewriter, node_warn, node_warn, "obj.warn_me")

  handle_pre_checks(rewriter, node_warn, node_attr_upd, "obj.warn_me")

  handle_pre_checks(rewriter, node1, node1, "f")

  class RewriterNoLifecycle(DummyRewriter):
    """Class doc."""

    def _get_source_lifecycle_lists(self) -> Tuple[Set[str], Set[str]]:
      """Function doc."""
      raise AttributeError()

  rewriter_no_lifecycle: RewriterNoLifecycle = RewriterNoLifecycle()
  try:
    handle_pre_checks(rewriter_no_lifecycle, node_attr, node_attr, "obj.strip_me")
  except AttributeError:
    pass

  rewriter._stateful.append("state_f")
  rewriter.semantics._configs["jax"] = {"stateful_call": {"method": "mock", "prepend_arg": "vars"}}
  handle_pre_checks(rewriter, node1, node1, "state_f")

  rewriter.semantics._configs["jax"] = {}
  handle_pre_checks(rewriter, node1, node1, "state_f")


def test_resolve_implicit_method_branches() -> None:
  """Docstring."""
  rewriter: DummyRewriter = DummyRewriter()

  node_attr: cst.Call = getattr(getattr(cst.parse_statement("obj.f()"), "body")[0], "value")
  resolve_implicit_method(rewriter, node_attr, "obj.f")

  node_name: cst.Call = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  resolve_implicit_method(rewriter, node_name, "f")

  node_self: cst.Call = getattr(getattr(cst.parse_statement("self.f()"), "body")[0], "value")
  resolve_implicit_method(rewriter, node_self, "self.f")

  node_mod: cst.Call = getattr(getattr(cst.parse_statement("alias.f()"), "body")[0], "value")
  resolve_implicit_method(rewriter, node_mod, "alias.f")

  class RewriterNoModuleAttr(DummyRewriter):
    """Class doc."""

    def _is_module_alias(self, node: cst.CSTNode) -> bool:
      """Function doc."""
      raise AttributeError()

  rewriter_no_attr_alias: RewriterNoModuleAttr = RewriterNoModuleAttr()
  try:
    resolve_implicit_method(rewriter_no_attr_alias, node_attr, "obj.f")
  except AttributeError:
    pass

  # Completely missing method
  class RewriterMissingModuleAttr(DummyRewriter):
    """Class doc."""

    def _is_module_alias(self, node: cst.CSTNode) -> bool:
      """Function doc."""
      raise AttributeError()

  rm_attr: RewriterMissingModuleAttr = RewriterMissingModuleAttr()
  try:
    resolve_implicit_method(rm_attr, node_attr, "obj.f")
  except AttributeError:
    pass

  class DummyType:
    """Class doc."""

    def __init__(self, n: str, fw: Optional[str] = None) -> None:
      """Init doc."""
      self.name = n
      if fw:
        self.framework = fw

  class RewriterNoAlias(DummyRewriter):
    """Class doc."""

    def _is_module_alias(self, node: cst.CSTNode) -> bool:
      """Function doc."""
      return False

  rewriter_no_alias: RewriterNoAlias = RewriterNoAlias()
  rewriter_no_alias.context.symbol_table.types["obj"] = DummyType("MyType")
  rewriter_no_alias._mappings["MyType.f"] = {"test": True}
  assert resolve_implicit_method(rewriter_no_alias, node_attr, "obj.f") == "MyType.f"

  # 159 -> 174 context missing symbol_table
  class ContextNoSym:
    """Class doc."""

    pass

  rewriter_no_alias.context = ContextNoSym()
  try:
    resolve_implicit_method(rewriter_no_alias, node_attr, "obj.f")
  except AttributeError:
    pass
  rewriter_no_alias.context = DummyContext()

  rewriter_no_alias.context.symbol_table.types["obj2"] = DummyType("Tensor")  # no framework
  resolve_implicit_method(
    rewriter_no_alias, getattr(getattr(cst.parse_statement("obj2.f()"), "body")[0], "value"), "obj2.f"
  )

  # Ensure mapping returns False to cover 170 -> 174
  rewriter_no_alias._mappings["MyType.g"] = None
  resolve_implicit_method(
    rewriter_no_alias, getattr(getattr(cst.parse_statement("obj.g()"), "body")[0], "value"), "obj.g"
  )

  rewriter_no_alias.context.symbol_table.types["t"] = DummyType("Tensor", fw="torch")
  node_tensor: cst.Call = getattr(getattr(cst.parse_statement("t.f()"), "body")[0], "value")
  rewriter_no_alias._mappings["torch.Tensor.f"] = {"test": True}
  assert resolve_implicit_method(rewriter_no_alias, node_tensor, "t.f") == "torch.Tensor.f"

  class RewriterNoMapping(DummyRewriter):
    """Class doc."""

    def _get_mapping(self, name: str, silent: bool = False) -> Any:
      """Function doc."""
      raise AttributeError()

  rnm: RewriterNoMapping = RewriterNoMapping()
  rnm.context.symbol_table.types["obj"] = DummyType("MyType")
  try:
    resolve_implicit_method(rnm, node_attr, "obj.f")
  except AttributeError:
    pass

  # No get_mapping on rewriter at all
  class RewriterMissingMappingAttr(DummyRewriter):
    """Class doc."""

    def _get_mapping(self, name: str, silent: bool = False) -> Any:
      """Function doc."""
      raise AttributeError()

  r_no_map: RewriterMissingMappingAttr = RewriterMissingMappingAttr()
  r_no_map.context.symbol_table.types["obj"] = DummyType("MyType")
  try:
    resolve_implicit_method(r_no_map, node_attr, "obj.f")
  except AttributeError:
    pass

  node_unknown: cst.Call = getattr(getattr(cst.parse_statement("unknown.f()"), "body")[0], "value")

  class DummyTargetRewriter(DummyRewriter):
    """Class doc."""

    pass

  tr: DummyTargetRewriter = DummyTargetRewriter()
  tr._mappings["torch.Tensor.f"] = {"test": True}
  assert resolve_implicit_method(tr, node_unknown, "unknown.f") == "torch.Tensor.f"

  # Not found in _mappings
  resolve_implicit_method(tr, getattr(getattr(cst.parse_statement("unknown.g()"), "body")[0], "value"), "unknown.g")

  class TargetNoMap(DummyTargetRewriter):
    """Class doc."""

    def _get_mapping(self, name: str, silent: bool = False) -> Any:
      """Function doc."""
      raise AttributeError()

  tr_no_map: TargetNoMap = TargetNoMap()
  try:
    resolve_implicit_method(tr_no_map, node_unknown, "unknown.f")
  except AttributeError:
    pass

  class TargetMissingMapAttr(DummyTargetRewriter):
    """Class doc."""

    def _get_mapping(self, name: str, silent: bool = False) -> Any:
      """Function doc."""
      raise AttributeError()

  tr_missing_map: TargetMissingMapAttr = TargetMissingMapAttr()
  try:
    resolve_implicit_method(tr_missing_map, node_unknown, "unknown.f")
  except AttributeError:
    pass

  tr_no_traits: DummyTargetRewriter = DummyTargetRewriter()
  del tr_no_traits.source_traits
  tr_no_traits.semantics._configs["torch"] = {"traits": {"implicit_method_roots": ["torch.Tensor"]}}
  resolve_implicit_method(tr_no_traits, node_unknown, "unknown.f")

  # 174 -> 194 (no _get_target_traits)
  class TargetNoTraitsAttr(DummyTargetRewriter):
    """Class doc."""

    def _get_target_traits(self) -> DummyTraits:
      """Function doc."""
      raise AttributeError()

  tnt: TargetNoTraitsAttr = TargetNoTraitsAttr()
  try:
    resolve_implicit_method(tnt, node_unknown, "unknown.f")
  except AttributeError:
    pass


def test_handle_pre_checks_no_hook() -> None:
  """Function doc."""
  from unittest.mock import MagicMock, patch

  import libcst as cst

  from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks

  class DummyRewriter:
    """Class doc."""

    def __init__(self) -> None:
      """Init doc."""
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
  """Function doc."""
  from unittest.mock import MagicMock

  import libcst as cst

  from ml_switcheroo.core.rewriter.calls.pre import resolve_implicit_method

  # Branch [154, 157]: Missing _is_module_alias
  class RewriterNoIsModule:
    """Class doc."""

    pass

  rewriter: RewriterNoIsModule = RewriterNoIsModule()
  node_attr: cst.Call = getattr(getattr(cst.parse_statement("obj.f()"), "body")[0], "value")
  resolve_implicit_method(rewriter, node_attr, "obj.f")

  # Branch [168, 174]: Has symbol table, but missing _get_mapping
  class SymType:
    """Class doc."""

    name: str = "Tensor"
    framework: str = "torch"

  class SymTable:
    """Class doc."""

    def get_type(self, node: cst.CSTNode) -> SymType:
      """Function doc."""
      return SymType()

  class RewriterSymTableNoMap:
    """Class doc."""

    def __init__(self) -> None:
      """Init doc."""
      self.context: MagicMock = MagicMock()
      self.context.symbol_table = SymTable()

  r2: RewriterSymTableNoMap = RewriterSymTableNoMap()
  resolve_implicit_method(r2, node_attr, "obj.f")

  # Branch [189, 187]: Fallback legacy, missing _get_mapping
  class RewriterLegacyNoMap:
    """Class doc."""

    def __init__(self) -> None:
      """Init doc."""
      self.source_traits: MagicMock = MagicMock()
      self.source_traits.implicit_method_roots = ["torch.Tensor", "numpy.ndarray"]

    def _get_target_traits(self) -> None:
      """Function doc."""
      pass

  r3: RewriterLegacyNoMap = RewriterLegacyNoMap()
  resolve_implicit_method(r3, node_attr, "obj.f")


# --- Merged from test_rewriter_calls_pre_extra.py ---


class DummyRewriterExtra:
  """Docstring."""

  def __init__(self) -> None:
    """Docstring."""
    self.context: MagicMock = MagicMock()
    self.context.hook_context = MagicMock()
    self.semantics: MagicMock = MagicMock()
    self.semantics.get_definition.return_value = None

  def _report_warning(self, w: str) -> None:
    """Docstring."""
    pass


@pytest.fixture(autouse=True)
def _cleanup() -> Generator[None, None, None]:
  """Docstring."""
  import ml_switcheroo.core.hooks_registry as hr

  hr.clear_hooks()
  hr._PLUGINS_LOADED = True
  yield
  hr.clear_hooks()


def test_handle_pre_checks_inplace_no_change() -> None:
  """Docstring."""
  rewriter: DummyRewriter = DummyRewriter()

  # 97-98: in-place unroll hook doesn't change node
  @register_hook("unroll_inplace_ops")
  def mock_hook(node: cst.CSTNode, ctx: MagicMock) -> cst.CSTNode:
    """Docstring."""
    return node

  original: cst.Call = getattr(getattr(cst.parse_statement("foo_()"), "body")[0], "value")
  res: Tuple[bool, cst.CSTNode] = handle_pre_checks(rewriter, original, original, "foo_")
  assert res[0] is False


def test_handle_pre_checks_inplace_change() -> None:
  """Docstring."""
  rewriter: DummyRewriter = DummyRewriter()

  # 97-98: in-place unroll hook changes node
  @register_hook("unroll_inplace_ops")
  def mock_hook(node: cst.CSTNode, ctx: MagicMock) -> cst.CSTNode:
    """Docstring."""
    return getattr(cst.parse_statement("b = 1"), "body")[0]

  original: cst.Call = getattr(getattr(cst.parse_statement("foo_()"), "body")[0], "value")
  res: Tuple[bool, cst.CSTNode] = handle_pre_checks(rewriter, original, original, "foo_")
  assert res[0] is True
