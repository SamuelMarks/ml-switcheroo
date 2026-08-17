"""Module docstring."""

import libcst as cst
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method
from ml_switcheroo.core.hooks_registry import register_hook, clear_hooks


class DummyTraits:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.functional_execution_method = "apply"
    self.implicit_method_roots = ["torch.Tensor"]


class DummySemantics:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self._defs = {}
    self._configs = {"torch": {"traits": {}}}

  def get_definition(self, name):
    """Docstring."""
    return self._defs.get(name)

  def get_framework_config(self, fw):
    """Docstring."""
    return self._configs.get(fw, {})


class DummySymbolTable:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.types = {}

  def get_type(self, node):
    """Docstring."""
    if isinstance(node, cst.Name):
      return self.types.get(node.value)
    return None


class DummyContext:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.hook_context = {}
    self.symbol_table = DummySymbolTable()


class DummyRewriter:
  """Docstring."""

  def __init__(self):
    """Docstring."""
    self.source_fw = "torch"
    self.target_fw = "jax"
    self.source_traits = DummyTraits()
    self.semantics = DummySemantics()
    self.context = DummyContext()
    self._mappings = {}
    self._stateful = []

  def _get_source_traits(self):
    """Function doc."""
    return getattr(self, "source_traits", DummyTraits())

  def _report_warning(self, msg):
    """Function doc."""
    pass

  def _get_mapping(self, name, silent=False):
    """Function doc."""
    return self._mappings.get(name)

  def _is_stateful(self, name):
    """Function doc."""
    return name in self._stateful

  def _get_source_lifecycle_lists(self):
    """Function doc."""
    return {"strip_me"}, {"warn_me"}

  def _is_module_alias(self, node):
    """Function doc."""
    return isinstance(node, cst.Name) and node.value == "alias"

  def _get_target_traits(self):
    """Function doc."""
    return DummyTraits()


def test_handle_pre_checks_branches():
  """Docstring."""
  rewriter = DummyRewriter()

  node1 = cst.parse_statement("f()").body[0].value
  handle_pre_checks(rewriter, node1, node1, "f")

  rewriter_no_attr = DummyRewriter()
  del rewriter_no_attr.source_traits
  handle_pre_checks(rewriter_no_attr, node1, node1, "f")

  class NoTraits:
    """Class doc."""

    def __init__(self):
      """Init doc."""
      self.semantics = DummySemantics()

    def _get_target_traits(self):
      """Function doc."""
      return DummyTraits()

  nt_rewriter = NoTraits()
  handle_pre_checks(nt_rewriter, node1, node1, "f")

  node_apply = cst.parse_statement("obj.apply(vars, x)").body[0].value
  handled, res = handle_pre_checks(rewriter, node_apply, node_apply, "obj.apply")
  assert handled

  node_apply_noargs = cst.parse_statement("obj.apply()").body[0].value
  handle_pre_checks(rewriter, node_apply_noargs, node_apply_noargs, "obj.apply")

  node_apply_orig = cst.parse_statement("obj.apply()").body[0].value
  node_apply_upd = cst.parse_statement("f()").body[0].value
  handle_pre_checks(rewriter, node_apply_orig, node_apply_upd, "obj.apply")

  # 69 -> 86 func_name is None
  handle_pre_checks(rewriter, node1, node1, None)

  rewriter._mappings["plugin_func"] = {"requires_plugin": True}
  handle_pre_checks(rewriter, node1, node1, "plugin_func")

  class RewriterNoMappingAttr(DummyRewriter):
    """Class doc."""

    def _get_mapping(self, name, silent=False):
      """Function doc."""
      raise AttributeError()

  rewriter_no_map_attr = RewriterNoMappingAttr()
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

  def dummy_hook(node, ctx):
    """Function doc."""
    return cst.Name("changed")

  register_hook("unroll_inplace_ops")(dummy_hook)
  handle_pre_checks(rewriter, node1, node1, "inplace_func")

  def dummy_hook_same(node, ctx):
    """Function doc."""
    return node

  clear_hooks()
  register_hook("unroll_inplace_ops")(dummy_hook_same)
  handle_pre_checks(rewriter, node1, node1, "inplace_func")

  clear_hooks()
  handle_pre_checks(rewriter, node1, node1, "inplace_func")

  handle_pre_checks(rewriter, node1, node1, "add_")

  handle_pre_checks(rewriter, node1, node1, "add")

  node_attr = cst.parse_statement("obj.strip_me()").body[0].value
  handle_pre_checks(rewriter, node_attr, node_attr, "obj.strip_me")

  node_attr_upd = cst.parse_statement("f()").body[0].value
  handle_pre_checks(rewriter, node_attr, node_attr_upd, "obj.strip_me")

  node_warn = cst.parse_statement("obj.warn_me()").body[0].value
  handle_pre_checks(rewriter, node_warn, node_warn, "obj.warn_me")

  handle_pre_checks(rewriter, node_warn, node_attr_upd, "obj.warn_me")

  handle_pre_checks(rewriter, node1, node1, "f")

  class RewriterNoLifecycle(DummyRewriter):
    """Class doc."""

    def _get_source_lifecycle_lists(self):
      """Function doc."""
      raise AttributeError()

  rewriter_no_lifecycle = RewriterNoLifecycle()
  try:
    handle_pre_checks(rewriter_no_lifecycle, node_attr, node_attr, "obj.strip_me")
  except AttributeError:
    pass

  rewriter._stateful.append("state_f")
  rewriter.semantics._configs["jax"] = {"stateful_call": {"method": "mock", "prepend_arg": "vars"}}
  handle_pre_checks(rewriter, node1, node1, "state_f")

  rewriter.semantics._configs["jax"] = {}
  handle_pre_checks(rewriter, node1, node1, "state_f")


def test_resolve_implicit_method_branches():
  """Docstring."""
  rewriter = DummyRewriter()

  node_attr = cst.parse_statement("obj.f()").body[0].value
  resolve_implicit_method(rewriter, node_attr, "obj.f")

  node_name = cst.parse_statement("f()").body[0].value
  resolve_implicit_method(rewriter, node_name, "f")

  node_self = cst.parse_statement("self.f()").body[0].value
  resolve_implicit_method(rewriter, node_self, "self.f")

  node_mod = cst.parse_statement("alias.f()").body[0].value
  resolve_implicit_method(rewriter, node_mod, "alias.f")

  class RewriterNoModuleAttr(DummyRewriter):
    """Class doc."""

    def _is_module_alias(self, node):
      """Function doc."""
      raise AttributeError()

  rewriter_no_attr_alias = RewriterNoModuleAttr()
  try:
    resolve_implicit_method(rewriter_no_attr_alias, node_attr, "obj.f")
  except AttributeError:
    pass

  # Completely missing method
  class RewriterMissingModuleAttr(DummyRewriter):
    """Class doc."""

    def _is_module_alias(self, node):
      """Function doc."""
      raise AttributeError()

  rm_attr = RewriterMissingModuleAttr()
  try:
    resolve_implicit_method(rm_attr, node_attr, "obj.f")
  except AttributeError:
    pass

  class DummyType:
    """Class doc."""

    def __init__(self, n, fw=None):
      """Init doc."""
      self.name = n
      if fw:
        self.framework = fw

  class RewriterNoAlias(DummyRewriter):
    """Class doc."""

    def _is_module_alias(self, node):
      """Function doc."""
      return False

  rewriter_no_alias = RewriterNoAlias()
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
  resolve_implicit_method(rewriter_no_alias, cst.parse_statement("obj2.f()").body[0].value, "obj2.f")

  # Ensure mapping returns False to cover 170 -> 174
  rewriter_no_alias._mappings["MyType.g"] = None
  resolve_implicit_method(rewriter_no_alias, cst.parse_statement("obj.g()").body[0].value, "obj.g")

  rewriter_no_alias.context.symbol_table.types["t"] = DummyType("Tensor", fw="torch")
  node_tensor = cst.parse_statement("t.f()").body[0].value
  rewriter_no_alias._mappings["torch.Tensor.f"] = {"test": True}
  assert resolve_implicit_method(rewriter_no_alias, node_tensor, "t.f") == "torch.Tensor.f"

  class RewriterNoMapping(DummyRewriter):
    """Class doc."""

    def _get_mapping(self, name, silent=False):
      """Function doc."""
      raise AttributeError()

  rnm = RewriterNoMapping()
  rnm.context.symbol_table.types["obj"] = DummyType("MyType")
  try:
    resolve_implicit_method(rnm, node_attr, "obj.f")
  except AttributeError:
    pass

  # No get_mapping on rewriter at all
  class RewriterMissingMappingAttr(DummyRewriter):
    """Class doc."""

    def _get_mapping(self, name, silent=False):
      """Function doc."""
      raise AttributeError()

  r_no_map = RewriterMissingMappingAttr()
  r_no_map.context.symbol_table.types["obj"] = DummyType("MyType")
  try:
    resolve_implicit_method(r_no_map, node_attr, "obj.f")
  except AttributeError:
    pass

  node_unknown = cst.parse_statement("unknown.f()").body[0].value

  class DummyTargetRewriter(DummyRewriter):
    """Class doc."""

    pass

  tr = DummyTargetRewriter()
  tr._mappings["torch.Tensor.f"] = {"test": True}
  assert resolve_implicit_method(tr, node_unknown, "unknown.f") == "torch.Tensor.f"

  # Not found in _mappings
  resolve_implicit_method(tr, cst.parse_statement("unknown.g()").body[0].value, "unknown.g")

  class TargetNoMap(DummyTargetRewriter):
    """Class doc."""

    def _get_mapping(self, name, silent=False):
      """Function doc."""
      raise AttributeError()

  tr_no_map = TargetNoMap()
  try:
    resolve_implicit_method(tr_no_map, node_unknown, "unknown.f")
  except AttributeError:
    pass

  class TargetMissingMapAttr(DummyTargetRewriter):
    """Class doc."""

    def _get_mapping(self, name, silent=False):
      """Function doc."""
      raise AttributeError()

  tr_missing_map = TargetMissingMapAttr()
  try:
    resolve_implicit_method(tr_missing_map, node_unknown, "unknown.f")
  except AttributeError:
    pass

  tr_no_traits = DummyTargetRewriter()
  del tr_no_traits.source_traits
  tr_no_traits.semantics._configs["torch"] = {"traits": {"implicit_method_roots": ["torch.Tensor"]}}
  resolve_implicit_method(tr_no_traits, node_unknown, "unknown.f")

  # 174 -> 194 (no _get_target_traits)
  class TargetNoTraitsAttr(DummyTargetRewriter):
    """Class doc."""

    def _get_target_traits(self):
      """Function doc."""
      raise AttributeError()

  tnt = TargetNoTraitsAttr()
  try:
    resolve_implicit_method(tnt, node_unknown, "unknown.f")
  except AttributeError:
    pass


def test_handle_pre_checks_no_hook():
  """Function doc."""
  from unittest.mock import patch, MagicMock
  import libcst as cst
  from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks

  class DummyRewriter:
    """Class doc."""

    def __init__(self):
      """Init doc."""
      self.semantics = MagicMock()
      self.semantics.get_definition.return_value = None

  rewriter = DummyRewriter()
  original = cst.parse_statement("f_()").body[0].value

  with patch("ml_switcheroo.core.rewriter.calls.pre.get_hook", return_value=None):
    res, updated = handle_pre_checks(rewriter, original, original, "f_")
    assert not res


def test_resolve_implicit_method_missing_attributes():
  """Function doc."""
  import libcst as cst
  from ml_switcheroo.core.rewriter.calls.pre import resolve_implicit_method
  from unittest.mock import MagicMock

  # Branch [154, 157]: Missing _is_module_alias
  class RewriterNoIsModule:
    """Class doc."""

    pass

  rewriter = RewriterNoIsModule()
  node_attr = cst.parse_statement("obj.f()").body[0].value
  resolve_implicit_method(rewriter, node_attr, "obj.f")

  # Branch [168, 174]: Has symbol table, but missing _get_mapping
  class SymType:
    """Class doc."""

    name = "Tensor"
    framework = "torch"

  class SymTable:
    """Class doc."""

    def get_type(self, node):
      """Function doc."""
      return SymType()

  class RewriterSymTableNoMap:
    """Class doc."""

    def __init__(self):
      """Init doc."""
      self.context = MagicMock()
      self.context.symbol_table = SymTable()

  r2 = RewriterSymTableNoMap()
  resolve_implicit_method(r2, node_attr, "obj.f")

  # Branch [189, 187]: Fallback legacy, missing _get_mapping
  class RewriterLegacyNoMap:
    """Class doc."""

    def __init__(self):
      """Init doc."""
      self.source_traits = MagicMock()
      self.source_traits.implicit_method_roots = ["torch.Tensor", "numpy.ndarray"]

    def _get_target_traits(self):
      """Function doc."""
      pass

  r3 = RewriterLegacyNoMap()
  resolve_implicit_method(r3, node_attr, "obj.f")
