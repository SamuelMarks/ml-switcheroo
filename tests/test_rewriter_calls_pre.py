"""Test suite for rewriter calls pre."""

from ml_switcheroo.enums import RuntimeTensorOrValue

import typing
from ml_switcheroo.core.hooks import HookContext
from pydantic import JsonValue

from unittest.mock import MagicMock

import libcst as cst
import pytest

from ml_switcheroo.core.hooks_registry import clear_hooks, register_hook
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method


class DummyTraits:
  """Test helper class DummyTraits."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.functional_execution_method = "apply"
    self.implicit_method_roots = ["torch.Tensor"]


class DummySemantics:
  """Test helper class DummySemantics."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self._defs: dict[str, RuntimeTensorOrValue] = {}
    self._configs: dict[str, RuntimeTensorOrValue] = {"torch": {"traits": {}}}

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Execute get definition helper logic.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    _ret_v: tuple[str, dict[str, JsonValue]] | None = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v")._defs}, "__getitem__")("v").get}, "__getitem__"
        )("v")(getattr({"v": name}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v

  def get_framework_config(self, fw: str) -> dict[str, JsonValue]:
    """Execute get framework config helper logic.

    Args:
        fw: Framework identifier string (e.g., torch, jax, mlx).

    Returns:
        Resulting `dict[str, JsonValue]` from get framework config.
    """
    _ret_v: dict[str, JsonValue] = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v")._configs}, "__getitem__")("v").get}, "__getitem__"
        )("v")(getattr({"v": fw}, "__getitem__")("v"), getattr({"v": {}}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v


class DummySymbolTable:
  """Test helper class DummySymbolTable."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.types: dict[str, RuntimeTensorOrValue] = {}

  def get_type(self, node: cst.CSTNode) -> str | None:
    """Execute get type helper logic.

    Args:
        node: CST or AST node under inspection or transformation.

    Returns:
        Resulting `str | None` from get type.
    """
    if isinstance(node, cst.Name):
      _ret_v: str | None = getattr(
        {
          "v": getattr(
            {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v").types}, "__getitem__")("v").get}, "__getitem__"
          )("v")(getattr({"v": getattr({"v": node}, "__getitem__")("v").value}, "__getitem__")("v"))
        },
        "__getitem__",
      )("v")
      return _ret_v
    return None


class DummyContext:
  """Test helper class DummyContext."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.hook_context: dict[str, RuntimeTensorOrValue] = {}
    self.symbol_table = DummySymbolTable()


class DummyRewriter:
  """Test helper class DummyRewriter."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.source_framework = "torch"
    self.target_framework = "jax"
    self.source_traits = DummyTraits()
    self.semantics = DummySemantics()
    self.context = DummyContext()
    self._mappings: dict[str, RuntimeTensorOrValue] = {}
    self._stateful: list[RuntimeTensorOrValue] = []

  def _get_source_traits(self) -> DummyTraits:
    """Execute get source traits helper logic.

    Returns:
        Resulting `DummyTraits` from get source traits.
    """
    return getattr(self, "source_traits", DummyTraits())

  def _report_warning(self, msg: str) -> None:
    """Execute report warning helper logic.

    Args:
        msg: The msg (str) for report warning.
    """
    pass

  def _get_mapping(self, name: str, silent: bool = False) -> dict[str, JsonValue] | None:
    """Execute get mapping helper logic.

    Args:
        name: Identifier or operation name string.
        silent: The silent (bool) for get mapping.

    Returns:
        Resulting `dict[str, JsonValue] | None` from get mapping.
    """
    _ret_v: dict[str, JsonValue] | None = getattr(
      {
        "v": getattr(
          {"v": getattr({"v": getattr({"v": self}, "__getitem__")("v")._mappings}, "__getitem__")("v").get}, "__getitem__"
        )("v")(getattr({"v": name}, "__getitem__")("v"))
      },
      "__getitem__",
    )("v")
    return _ret_v

  def _is_stateful(self, name: str) -> bool:
    """Execute is stateful helper logic.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `bool` from is stateful.
    """
    return name in self._stateful

  def _get_source_lifecycle_lists(self) -> tuple[set[str], set[str]]:
    """Execute get source lifecycle lists helper logic.

    Returns:
        Resulting `tuple[set[str], set[str]]` from get source lifecycle lists.
    """
    return {"strip_me"}, {"warn_me"}

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """Execute is module alias helper logic.

    Args:
        node: CST or AST node under inspection or transformation.

    Returns:
        Resulting `bool` from is module alias.
    """
    return isinstance(node, cst.Name) and node.value == "alias"

  def _get_target_traits(self) -> DummyTraits:
    """Execute get target traits helper logic.

    Returns:
        Resulting `DummyTraits` from get target traits.
    """
    return DummyTraits()


def test_handle_pre_checks_branches() -> None:
  """Verify handle pre checks branches behavior."""
  rewriter = DummyRewriter()

  node1 = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node1, node1, "f") == (False, node1)

  rewriter_no_attr = DummyRewriter()
  del rewriter_no_attr.source_traits
  assert handle_pre_checks(rewriter_no_attr, node1, node1, "f") == (False, node1)

  class NoTraits:
    """Test helper class NoTraits."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.semantics = DummySemantics()

    def _get_target_traits(self) -> DummyTraits:
      """Execute get target traits helper logic.

      Returns:
          Resulting `DummyTraits` from get target traits.
      """
      return DummyTraits()

  nt_rewriter = NoTraits()
  assert getattr(
    {
      "v": getattr({"v": handle_pre_checks}, "__getitem__")("v")(
        getattr({"v": nt_rewriter}, "__getitem__")("v"),
        getattr({"v": node1}, "__getitem__")("v"),
        getattr({"v": node1}, "__getitem__")("v"),
        getattr({"v": "f"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") == getattr(
    {"v": (getattr({"v": False}, "__getitem__")("v"), getattr({"v": node1}, "__getitem__")("v"))}, "__getitem__"
  )("v")

  node_apply = getattr(getattr(cst.parse_statement("obj.apply(vars, x)"), "body")[0], "value")
  handled, res = handle_pre_checks(rewriter, node_apply, node_apply, "obj.apply")
  assert handled is True
  assert isinstance(res, cst.Call)

  node_apply_noargs = getattr(getattr(cst.parse_statement("obj.apply()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node_apply_noargs, node_apply_noargs, "obj.apply")[0] is True

  node_apply_orig = getattr(getattr(cst.parse_statement("obj.apply()"), "body")[0], "value")
  node_apply_upd = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node_apply_orig, node_apply_upd, "obj.apply")[0] is False

  # 69 -> 86 func_name is None
  assert handle_pre_checks(rewriter, node1, node1, None) == (False, node1)

  rewriter._mappings["plugin_func"] = {"requires_plugin": True}
  assert handle_pre_checks(rewriter, node1, node1, "plugin_func") == (False, node1)

  class RewriterNoMappingAttr(DummyRewriter):
    """Test helper class RewriterNoMappingAttr."""

    def _get_mapping(self, name: str, silent: bool = False) -> dict[str, JsonValue] | None:
      """Execute get mapping helper logic.

      Args:
          name: Identifier or operation name string.
          silent: The silent (bool) for get mapping.

      Returns:
          Resulting `dict[str, JsonValue] | None` from get mapping.
      """
      raise AttributeError()

  rewriter_no_map_attr = RewriterNoMappingAttr()
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

  def dummy_hook(node: cst.CSTNode, ctx: HookContext) -> cst.CSTNode:
    """Execute dummy hook helper logic.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext instance under test.

    Returns:
        Resulting `cst.CSTNode` from dummy hook.
    """
    return cst.Name("changed")

  register_hook("unroll_inplace_ops")(dummy_hook)
  assert handle_pre_checks(rewriter, node1, node1, "inplace_func")[0] is True

  def dummy_hook_same(node: cst.CSTNode, ctx: HookContext) -> cst.CSTNode:
    """Execute dummy hook same helper logic.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext instance under test.

    Returns:
        Resulting `cst.CSTNode` from dummy hook same.
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

  node_attr = getattr(getattr(cst.parse_statement("obj.strip_me()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node_attr, node_attr, "obj.strip_me")[0] is True

  node_attr_upd = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node_attr, node_attr_upd, "obj.strip_me")[0] is False

  node_warn = getattr(getattr(cst.parse_statement("obj.warn_me()"), "body")[0], "value")
  assert handle_pre_checks(rewriter, node_warn, node_warn, "obj.warn_me")[0] is True

  assert handle_pre_checks(rewriter, node_warn, node_attr_upd, "obj.warn_me")[0] is False

  assert handle_pre_checks(rewriter, node1, node1, "f")[0] is False

  class RewriterNoLifecycle(DummyRewriter):
    """Test helper class RewriterNoLifecycle."""

    def _get_source_lifecycle_lists(self) -> tuple[set[str], set[str]]:
      """Execute get source lifecycle lists helper logic.

      Returns:
          Resulting `tuple[set[str], set[str]]` from get source lifecycle lists.
      """
      raise AttributeError()

  rewriter_no_lifecycle = RewriterNoLifecycle()
  with pytest.raises(AttributeError):
    handle_pre_checks(rewriter_no_lifecycle, node_attr, node_attr, "obj.strip_me")

  rewriter._stateful.append("state_f")
  rewriter.semantics._configs["jax"] = {"stateful_call": {"method": "mock", "prepend_arg": "vars"}}
  assert handle_pre_checks(rewriter, node1, node1, "state_f")[0] is True

  rewriter.semantics._configs["jax"] = {}
  assert handle_pre_checks(rewriter, node1, node1, "state_f")[0] is False


def test_resolve_implicit_method_branches() -> None:
  """Verify resolve implicit method branches behavior."""
  rewriter = DummyRewriter()

  node_attr = getattr(getattr(cst.parse_statement("obj.f()"), "body")[0], "value")
  assert resolve_implicit_method(rewriter, node_attr, "obj.f") is None

  node_name = getattr(getattr(cst.parse_statement("f()"), "body")[0], "value")
  assert resolve_implicit_method(rewriter, node_name, "f") is None

  node_self = getattr(getattr(cst.parse_statement("self.f()"), "body")[0], "value")
  assert resolve_implicit_method(rewriter, node_self, "self.f") is None

  node_mod = getattr(getattr(cst.parse_statement("alias.f()"), "body")[0], "value")
  assert resolve_implicit_method(rewriter, node_mod, "alias.f") is None

  class RewriterNoModuleAttr(DummyRewriter):
    """Test helper class RewriterNoModuleAttr."""

    def _is_module_alias(self, node: cst.CSTNode) -> bool:
      """Execute is module alias helper logic.

      Args:
          node: CST or AST node under inspection or transformation.

      Returns:
          Resulting `bool` from is module alias.
      """
      raise AttributeError()

  rewriter_no_attr_alias = RewriterNoModuleAttr()
  with pytest.raises(AttributeError):
    resolve_implicit_method(rewriter_no_attr_alias, node_attr, "obj.f")

  # Completely missing method
  class RewriterMissingModuleAttr(DummyRewriter):
    """Test helper class RewriterMissingModuleAttr."""

    def _is_module_alias(self, node: cst.CSTNode) -> bool:
      """Execute is module alias helper logic.

      Args:
          node: CST or AST node under inspection or transformation.

      Returns:
          Resulting `bool` from is module alias.
      """
      raise AttributeError()

  rm_attr = RewriterMissingModuleAttr()
  with pytest.raises(AttributeError):
    resolve_implicit_method(rm_attr, node_attr, "obj.f")

  class DummyType:
    """Test helper class DummyType."""

    def __init__(self, n: str, fw: str | None = None) -> None:
      """Initialize test helper instance.

      Args:
          n: The n (str) for init.
          fw: Framework identifier string (e.g., torch, jax, mlx).
      """
      self.name = n
      if fw:
        self.framework = fw

  class RewriterNoAlias(DummyRewriter):
    """Test helper class RewriterNoAlias."""

    def _is_module_alias(self, node: cst.CSTNode) -> bool:
      """Execute is module alias helper logic.

      Args:
          node: CST or AST node under inspection or transformation.

      Returns:
          Resulting `bool` from is module alias.
      """
      return False

  rewriter_no_alias = RewriterNoAlias()
  rewriter_no_alias.context.symbol_table.types["obj"] = DummyType("MyType")
  rewriter_no_alias._mappings["MyType.f"] = {"test": True}
  assert resolve_implicit_method(rewriter_no_alias, node_attr, "obj.f") == "MyType.f"

  # 159 -> 174 context missing symbol_table
  class ContextNoSym:
    """Test helper class ContextNoSym."""

    pass

  setattr(
    getattr({"v": rewriter_no_alias}, "__getitem__")("v"),
    "context",
    getattr({"v": getattr({"v": ContextNoSym}, "__getitem__")("v")()}, "__getitem__")("v"),
  )
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
  node_tensor = getattr(getattr(cst.parse_statement("t.f()"), "body")[0], "value")
  rewriter_no_alias._mappings["torch.Tensor.f"] = {"test": True}
  assert resolve_implicit_method(rewriter_no_alias, node_tensor, "t.f") == "torch.Tensor.f"

  class RewriterNoMapping(DummyRewriter):
    """Test helper class RewriterNoMapping."""

    def _get_mapping(self, name: str, silent: bool = False) -> dict[str, JsonValue] | None:
      """Execute get mapping helper logic.

      Args:
          name: Identifier or operation name string.
          silent: The silent (bool) for get mapping.

      Returns:
          Resulting `dict[str, JsonValue] | None` from get mapping.
      """
      raise AttributeError()

  rnm = RewriterNoMapping()
  rnm.context.symbol_table.types["obj"] = DummyType("MyType")
  with pytest.raises(AttributeError):
    resolve_implicit_method(rnm, node_attr, "obj.f")

  # No get_mapping on rewriter at all
  class RewriterMissingMappingAttr(DummyRewriter):
    """Test helper class RewriterMissingMappingAttr."""

    def _get_mapping(self, name: str, silent: bool = False) -> dict[str, JsonValue] | None:
      """Execute get mapping helper logic.

      Args:
          name: Identifier or operation name string.
          silent: The silent (bool) for get mapping.

      Returns:
          Resulting `dict[str, JsonValue] | None` from get mapping.
      """
      raise AttributeError()

  r_no_map = RewriterMissingMappingAttr()
  r_no_map.context.symbol_table.types["obj"] = DummyType("MyType")
  with pytest.raises(AttributeError):
    resolve_implicit_method(r_no_map, node_attr, "obj.f")

  node_unknown = getattr(getattr(cst.parse_statement("unknown.f()"), "body")[0], "value")

  class DummyTargetRewriter(DummyRewriter):
    """Test helper class DummyTargetRewriter."""

    pass

  tr = DummyTargetRewriter()
  tr._mappings["torch.Tensor.f"] = {"test": True}
  assert resolve_implicit_method(tr, node_unknown, "unknown.f") == "torch.Tensor.f"

  # Not found in _mappings
  assert (
    resolve_implicit_method(tr, getattr(getattr(cst.parse_statement("unknown.g()"), "body")[0], "value"), "unknown.g")
    is None
  )

  class TargetNoMap(DummyTargetRewriter):
    """Test helper class TargetNoMap."""

    def _get_mapping(self, name: str, silent: bool = False) -> dict[str, JsonValue] | None:
      """Execute get mapping helper logic.

      Args:
          name: Identifier or operation name string.
          silent: The silent (bool) for get mapping.

      Returns:
          Resulting `dict[str, JsonValue] | None` from get mapping.
      """
      raise AttributeError()

  tr_no_map = TargetNoMap()
  with pytest.raises(AttributeError):
    resolve_implicit_method(tr_no_map, node_unknown, "unknown.f")

  class TargetMissingMapAttr(DummyTargetRewriter):
    """Test helper class TargetMissingMapAttr."""

    def _get_mapping(self, name: str, silent: bool = False) -> dict[str, JsonValue] | None:
      """Execute get mapping helper logic.

      Args:
          name: Identifier or operation name string.
          silent: The silent (bool) for get mapping.

      Returns:
          Resulting `dict[str, JsonValue] | None` from get mapping.
      """
      raise AttributeError()

  tr_missing_map = TargetMissingMapAttr()
  with pytest.raises(AttributeError):
    resolve_implicit_method(tr_missing_map, node_unknown, "unknown.f")

  tr_no_traits = DummyTargetRewriter()
  del tr_no_traits.source_traits
  tr_no_traits.semantics._configs["torch"] = {"traits": {"implicit_method_roots": ["torch.Tensor"]}}
  assert resolve_implicit_method(tr_no_traits, node_unknown, "unknown.f") is None

  # 174 -> 194 (no _get_target_traits)
  class TargetNoTraitsAttr(DummyTargetRewriter):
    """Test helper class TargetNoTraitsAttr."""

    def _get_target_traits(self) -> DummyTraits:
      """Execute get target traits helper logic.

      Returns:
          Resulting `DummyTraits` from get target traits.
      """
      raise AttributeError()

  tnt = TargetNoTraitsAttr()
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
      self.semantics = MagicMock()
      self.semantics.get_definition.return_value = None

  rewriter = DummyRewriter()
  original = getattr(getattr(cst.parse_statement("f_()"), "body")[0], "value")

  with patch("ml_switcheroo.core.rewriter.calls.pre.get_hook", return_value=None):
    (res, updated) = getattr(
      {
        "v": getattr({"v": handle_pre_checks}, "__getitem__")("v")(
          getattr({"v": rewriter}, "__getitem__")("v"),
          getattr({"v": original}, "__getitem__")("v"),
          getattr({"v": original}, "__getitem__")("v"),
          getattr({"v": "f_"}, "__getitem__")("v"),
        )
      },
      "__getitem__",
    )("v")
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

  rewriter = RewriterNoIsModule()
  node_attr = getattr(getattr(cst.parse_statement("obj.f()"), "body")[0], "value")
  assert getattr(
    {
      "v": getattr({"v": resolve_implicit_method}, "__getitem__")("v")(
        getattr({"v": rewriter}, "__getitem__")("v"),
        getattr({"v": node_attr}, "__getitem__")("v"),
        getattr({"v": "obj.f"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") is getattr({"v": None}, "__getitem__")("v")

  # Branch [168, 174]: Has symbol table, but missing _get_mapping
  class SymType:
    """Test helper class SymType."""

    name = "Tensor"
    framework = "torch"

  class SymTable:
    """Test helper class SymTable."""

    def get_type(self, node: cst.CSTNode) -> SymType:
      """Execute get type helper logic.

      Args:
          node: CST or AST node under inspection or transformation.

      Returns:
          Resulting `SymType` from get type.
      """
      return SymType()

  class RewriterSymTableNoMap:
    """Test helper class RewriterSymTableNoMap."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.context = MagicMock()
      self.context.symbol_table = SymTable()

  r2 = RewriterSymTableNoMap()
  assert getattr(
    {
      "v": getattr({"v": resolve_implicit_method}, "__getitem__")("v")(
        getattr({"v": r2}, "__getitem__")("v"),
        getattr({"v": node_attr}, "__getitem__")("v"),
        getattr({"v": "obj.f"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") is getattr({"v": None}, "__getitem__")("v")

  # Branch [189, 187]: Fallback legacy, missing _get_mapping
  class RewriterLegacyNoMap:
    """Test helper class RewriterLegacyNoMap."""

    def __init__(self) -> None:
      """Initialize test helper instance."""
      self.source_traits = MagicMock()
      self.source_traits.implicit_method_roots = ["torch.Tensor", "numpy.ndarray"]

    def _get_target_traits(self) -> None:
      """Execute get target traits helper logic."""
      pass

  r3 = RewriterLegacyNoMap()
  assert getattr(
    {
      "v": getattr({"v": resolve_implicit_method}, "__getitem__")("v")(
        getattr({"v": r3}, "__getitem__")("v"),
        getattr({"v": node_attr}, "__getitem__")("v"),
        getattr({"v": "obj.f"}, "__getitem__")("v"),
      )
    },
    "__getitem__",
  )("v") is getattr({"v": None}, "__getitem__")("v")


class DummyRewriterExtra:
  """Test helper class DummyRewriterExtra."""

  def __init__(self) -> None:
    """Initialize test helper instance."""
    self.context = MagicMock()
    self.context.hook_context = MagicMock()
    self.semantics = MagicMock()
    self.semantics.get_definition.return_value = None

  def _report_warning(self, w: str) -> None:
    """Execute report warning helper logic.

    Args:
        w: The w (str) for report warning.
    """
    pass


@pytest.fixture(autouse=True)
def _cleanup() -> typing.Generator[None, None, None]:
  """Execute cleanup helper logic.

  Returns:
      Configured `_cleanup` fixture value (typing.Generator[None, None, None]).
  """
  import ml_switcheroo.core.hooks_registry as hr

  hr.clear_hooks()
  hr._PLUGINS_LOADED = True
  yield
  hr.clear_hooks()


def test_handle_pre_checks_inplace_no_change() -> None:
  """Verify handle pre checks inplace no change behavior."""
  rewriter = DummyRewriter()

  # 97-98: in-place unroll hook doesn't change node
  @register_hook("unroll_inplace_ops")
  def mock_hook(node: cst.CSTNode, ctx: HookContext) -> cst.CSTNode:
    """Execute mock hook helper logic.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext instance under test.

    Returns:
        Resulting `cst.CSTNode` from mock hook.
    """
    return node

  original = getattr(getattr(cst.parse_statement("foo_()"), "body")[0], "value")
  res = handle_pre_checks(rewriter, original, original, "foo_")
  assert res[0] is False


def test_handle_pre_checks_inplace_change() -> None:
  """Verify handle pre checks inplace change behavior."""
  rewriter = DummyRewriter()

  # 97-98: in-place unroll hook changes node
  @register_hook("unroll_inplace_ops")
  def mock_hook(node: cst.CSTNode, ctx: HookContext) -> cst.CSTNode:
    """Execute mock hook helper logic.

    Args:
        node: CST or AST node under inspection or transformation.
        ctx: HookContext or RewriterContext instance under test.

    Returns:
        Resulting `cst.CSTNode` from mock hook.
    """
    return cst.Name("replaced_expr")

  original = getattr(getattr(cst.parse_statement("foo_()"), "body")[0], "value")
  res = handle_pre_checks(rewriter, original, original, "foo_")
  assert res[0] is True
