"""Test suite for the Api Call Mixin module."""

import libcst as cst
import pytest
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.passes.api_call_mixin import CallRewriteHandler
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits


class MockTracer:
  """Test double `MockTracer` implementing `log_inspection`, `log_mutation` for API call mixin tests."""

  def log_inspection(self, *args: JsonValue, **kwargs: JsonValue) -> None:
    """Record a mock inspection trace event.

    Args:
        args: Positional argument specifications (`JsonValue`).
        kwargs: Keyword argument bindings (`JsonValue`).
    """

  def log_mutation(self, *args: JsonValue, **kwargs: JsonValue) -> None:
    """Record a mock mutation trace event.

    Args:
        args: Positional argument specifications (`JsonValue`).
        kwargs: Keyword argument bindings (`JsonValue`).
    """


class MockSemantics(SemanticsManager):
  """Test double `MockSemantics` subclassing `SemanticsManager` for API call mixin tests."""

  def __init__(self) -> None:
    """Initialize `MockSemantics` without loading disk defaults."""
    super().__init__(load_defaults=False)
    del self._key_origins

  def get_definition(self, name: str) -> tuple[str, dict[str, JsonValue]] | None:
    """Look up mock abstract operation definitions by qualified API name.

    Args:
        name: Identifier or operation name string.

    Returns:
        Resulting `tuple[str, dict[str, JsonValue]] | None` from get definition.
    """
    if name == "torch.known":
      return ("known_id", {})
    if name == "torch.nodep":
      return ("nodep_id", {})
    return None


class MockTransformer(cst.CSTTransformer):
  """Minimal `RewriterHost`-compliant CST transformer driving `CallRewriteHandler` in unit tests."""

  def __init__(self) -> None:
    """Initialize `MockTransformer` with `MockSemantics`, `RewriterContext`, and `CallRewriteHandler`."""
    super().__init__()
    self._semantics = MockSemantics()
    self._config = RuntimeConfig(source_framework="torch", target_framework="jax", strict_mode=True)
    self.context = RewriterContext(semantics=self._semantics, config=self._config)
    self._target_fw = "jax"
    self.warnings: list[str] = []
    self.failures: list[str] = []
    self._handler = CallRewriteHandler(self)

  @property
  def semantics(self) -> MockSemantics:
    """Return the bound `MockSemantics` instance.

    Returns:
        Bound `MockSemantics` instance.
    """
    return self._semantics

  @property
  def config(self) -> RuntimeConfig:
    """Return the bound `RuntimeConfig` instance.

    Returns:
        Bound `RuntimeConfig` instance.
    """
    return self._config

  @property
  def ctx(self) -> HookContext:
    """Return the `HookContext` bound to `self.context`.

    Returns:
        Bound `HookContext` instance.
    """
    return self.context.hook_context

  @property
  def source_framework(self) -> str:
    """Return the source framework identifier.

    Returns:
        Source framework string.
    """
    return "torch"

  @property
  def target_framework(self) -> str:
    """Return the target framework identifier.

    Returns:
        Target framework string.
    """
    return self._target_fw

  @target_framework.setter
  def target_framework(self, value: str) -> None:
    """Update the target framework identifier.

    Args:
        value: Replacement target framework string.
    """
    self._target_fw = value

  @property
  def strict_mode(self) -> bool:
    """Return whether strict mode is active.

    Returns:
        `True` for strict call checking.
    """
    return True

  @property
  def source_traits(self) -> StructuralTraits:
    """Return default source `StructuralTraits`.

    Returns:
        Default `StructuralTraits` instance.
    """
    return StructuralTraits()

  @property
  def target_traits(self) -> StructuralTraits:
    """Return default target `StructuralTraits`.

    Returns:
        Default `StructuralTraits` instance.
    """
    return StructuralTraits()

  def _get_target_traits(self) -> StructuralTraits:
    """Return default target `StructuralTraits`.

    Returns:
        Default `StructuralTraits` instance.
    """
    return StructuralTraits()

  def _get_source_lifecycle_lists(self) -> tuple[set[str], set[str]]:
    """Return empty lifecycle strip and warning sets.

    Returns:
        Tuple of empty sets.
    """
    return set(), set()

  def _get_qualified_name(self, node: cst.BaseExpression) -> str | None:
    """Resolve `cst.Name` or single-hop `cst.Attribute` to a qualified string.

    Args:
        node: CST expression node to inspect.

    Returns:
        Qualified API string or `None`.
    """
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.Attribute) and isinstance(node.value, cst.Name):
      return f"{node.value.value}.{node.attr.value}"
    return None

  def _get_mapping(self, name: str, silent: bool = False) -> dict[str, JsonValue] | None:
    """Return stubbed target variant mappings for known test API names.

    Args:
        name: Qualified API name string.
        silent: Whether to suppress missing-mapping failure diagnostics.

    Returns:
        Variant mapping dictionary or `None`.
    """
    if name == "torch.known":
      return {"api": "jnp.known"}
    if name == "torch.nodep":
      return {"api": "jnp.nodep", "min_version": "1.0", "max_version": "2.0"}
    if name == "torch.missing_def":
      return {"api": "jnp.missing_def"}
    return None

  def check_version_constraints(self, min_v: str | None, max_v: str | None) -> str | None:
    """Simulate version constraint warning when `min_v == '1.0'`.

    Args:
        min_v: Optional minimum version string.
        max_v: Optional maximum version string.

    Returns:
        Warning message or `None`.
    """
    if min_v == "1.0":
      return "Version warning"
    return None

  def _report_warning(self, reason: str) -> None:
    """Append `reason` to `self.warnings`.

    Args:
        reason: Warning message string.
    """
    self.warnings.append(reason)

  def _report_failure(self, reason: str) -> None:
    """Append `reason` to `self.failures`.

    Args:
        reason: Failure message string.
    """
    self.failures.append(reason)

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """Return `False` for module alias checks in isolated tests.

    Args:
        node: CST node under inspection.

    Returns:
        Always `False`.
    """
    return False

  def _handle_variant_imports(self, variant: dict[str, JsonValue]) -> None:
    """No-op import handler for isolated call mixin tests.

    Args:
        variant: Variant mapping dictionary.
    """

  def _create_name_node(self, api_path: str) -> cst.Name | cst.Attribute:
    """Construct a `cst.Name` node for `api_path`.

    Args:
        api_path: API identifier string.

    Returns:
        Constructed `cst.Name` node.
    """
    return cst.Name(api_path)

  def _create_dotted_name(self, name_str: str) -> cst.Name | cst.Attribute:
    """Construct a `cst.Name` node for `name_str`.

    Args:
        name_str: API identifier string.

    Returns:
        Constructed `cst.Name` node.
    """
    return cst.Name(name_str)

  def _mark_stateful(self, var_name: str) -> None:
    """No-op stateful variable marker.

    Args:
        var_name: Variable name string.
    """

  def _is_stateful(self, var_name: str) -> bool:
    """Return `False` for stateful variable checks.

    Args:
        var_name: Candidate variable name.

    Returns:
        Always `False`.
    """
    return False

  def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
    """Delegate `leave_Call` to `self._handler.rewrite_call`.

    Args:
        original_node: Original `cst.Call` node.
        updated_node: Updated `cst.Call` node.

    Returns:
        Rewritten `cst.BaseExpression`.
    """
    return self._handler.rewrite_call(original_node, updated_node)


def _extract_first_call(tree: cst.Module) -> cst.Call:
  """Extract the `cst.Call` expression from the first statement of `tree`.

  Args:
      tree: Rewritten LibCST module.

  Returns:
      Extracted `cst.Call` node.
  """
  stmt = cst.ensure_type(tree.body[0], cst.SimpleStatementLine)
  expr = cst.ensure_type(stmt.body[0], cst.Expr)
  return cst.ensure_type(expr.value, cst.Call)


def test_leave_Call_pre_check() -> None:
  """Verify `CallRewriteHandler` returns the pre-check replacement expression when handled."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "handle_pre_checks", lambda *args: (True, cst.Call(func=cst.Name("handled"), args=[])))
    tree = cst.parse_module("foo()")
    new_tree = tree.visit(transformer)
    call_node = _extract_first_call(new_tree)
    assert cst.ensure_type(call_node.func, cst.Name).value == "handled"


def test_leave_Call_no_mapping_super() -> None:
  """Verify `super()` calls bypass strict unmapped-API failure reporting."""
  transformer = MockTransformer()
  tree = cst.parse_module("super()")
  new_tree = tree.visit(transformer)
  call_node = _extract_first_call(new_tree)
  assert cst.ensure_type(call_node.func, cst.Name).value == "super"


def test_leave_Call_no_mapping_strict() -> None:
  """Verify unmapped calls in strict mode record a missing-semantics failure."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "get_tracer", lambda: MockTracer())
    m.setattr(mixin, "handle_pre_checks", lambda *args: (False, args[2]))
    tree = cst.parse_module("torch.unknown()")
    tree.visit(transformer)
    assert len(transformer.failures) == 1
    assert "not found in semantics" in transformer.failures[0]


def test_leave_Call_no_mapping_strict_neural(monkeypatch: pytest.MonkeyPatch) -> None:
  """Verify mapping a neural-tier abstraction to a math-only target reports a neural error.

  Args:
      monkeypatch: Pytest monkeypatch fixture.
  """
  transformer = MockTransformer()
  monkeypatch.setattr(transformer, "target_framework", "NumPy")
  transformer.semantics._key_origins = {"Linear": "neural"}
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "get_tracer", lambda: MockTracer())
    m.setattr(mixin, "handle_pre_checks", lambda *args: (False, args[2]))
    tree = cst.parse_module("torch.Linear(10, 20)")
    tree.visit(transformer)
    assert len(transformer.failures) == 1
    assert "Cannot map neural network abstraction" in transformer.failures[0]


def test_leave_Call_with_mapping() -> None:
  """Verify mapped calls invoke `execute_strategy` without emitting version warnings."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "execute_strategy", lambda *args: cst.Call(func=cst.Name("strategy_ok"), args=[]))
    m.setattr(mixin, "handle_post_processing", lambda s, n, *args: n)
    m.setattr(mixin, "handle_pre_checks", lambda *args: (False, args[2]))
    m.setattr(mixin, "log_diff", lambda *args: None)
    tree = cst.parse_module("torch.known()")
    new_tree = tree.visit(transformer)
    assert len(transformer.warnings) == 0
    call_node = _extract_first_call(new_tree)
    assert cst.ensure_type(call_node.func, cst.Name).value == "strategy_ok"


def test_leave_Call_with_mapping_version_warn() -> None:
  """Verify mapped calls with violated version constraints record a version warning."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "execute_strategy", lambda *args: cst.Call(func=cst.Name("strategy_ok"), args=[]))
    m.setattr(mixin, "handle_post_processing", lambda s, n, *args: n)
    m.setattr(mixin, "handle_pre_checks", lambda *args: (False, args[2]))
    m.setattr(mixin, "log_diff", lambda *args: None)
    tree = cst.parse_module("torch.nodep()")
    new_tree = tree.visit(transformer)
    assert len(transformer.warnings) == 1
    assert transformer.warnings[0] == "Version warning"
    call_node = _extract_first_call(new_tree)
    assert cst.ensure_type(call_node.func, cst.Name).value == "strategy_ok"


def test_leave_Call_implicit_method() -> None:
  """Verify implicit tensor method calls resolve and rewrite via `resolve_implicit_method`."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "resolve_implicit_method", lambda *args: "torch.nodep")
    m.setattr(mixin, "handle_pre_checks", lambda *args: (False, args[2]))
    m.setattr(mixin, "execute_strategy", lambda *args: cst.Call(func=cst.Name("implicit_ok"), args=[]))
    m.setattr(mixin, "handle_post_processing", lambda s, n, *args: n)
    m.setattr(mixin, "log_diff", lambda *args: None)
    tree = cst.parse_module("x.nodep()")
    new_tree = tree.visit(transformer)
    call_node = _extract_first_call(new_tree)
    assert cst.ensure_type(call_node.func, cst.Name).value == "implicit_ok"


def test_leave_Call_missing_def() -> None:
  """Verify mapped APIs lacking an abstract definition return the original call node unchanged."""
  transformer = MockTransformer()
  with pytest.MonkeyPatch().context() as m:
    import ml_switcheroo.core.rewriter.passes.api_call_mixin as mixin

    m.setattr(mixin, "handle_pre_checks", lambda *args: (False, args[2]))
    tree = cst.parse_module("torch.missing_def()")
    new_tree = tree.visit(transformer)
    call_node = _extract_first_call(new_tree)
    assert cst.ensure_type(call_node.func, cst.Attribute).attr.value == "missing_def"


def test_docstrings_are_not_placeholders() -> None:
  """Verify docstrings on ``api_call_mixin`` and ``CallRewriteHandler`` are substantive."""
  import ml_switcheroo.core.rewriter.passes.api_call_mixin as call_mod

  assert call_mod.__doc__ is not None
  assert "CallRewriteHandler" in call_mod.__doc__
  assert "Docstring" not in call_mod.__doc__

  cls_doc = call_mod.CallRewriteHandler.__doc__
  assert cls_doc is not None
  assert "Docstring" not in cls_doc
  assert "RewriterHost" in cls_doc or "handler" in cls_doc.lower()

  method_doc = call_mod.CallRewriteHandler.rewrite_call.__doc__
  assert method_doc is not None
  assert "cst.Call" in method_doc
  assert "Args:" in method_doc
  assert "Returns:" in method_doc
