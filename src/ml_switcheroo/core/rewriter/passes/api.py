"""API Logic Pass.

Provides ``ApiPass`` and ``ApiTransformer``, a single-inheritance ``cst.CSTTransformer``
composing ``CallRewriteHandler``, ``AttributeRewriteHandler``, and ``MappingResolver``
over a typed ``ApiPassState`` container.
"""

from __future__ import annotations

import typing

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.interface import RewriterPass
from ml_switcheroo.core.rewriter.passes.api_attr_mixin import AttributeRewriteHandler
from ml_switcheroo.core.rewriter.passes.api_call_mixin import CallRewriteHandler
from ml_switcheroo.core.rewriter.passes.api_helpers import ApiPassState, MappingResolver
from ml_switcheroo.core.rewriter.types import SignatureContext
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.schema import StructuralTraits

if typing.TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager


class ApiPass(RewriterPass):
  """Transformation pass for rewiring API usage across frameworks."""

  def transform(self, module: cst.Module, context: RewriterContext) -> cst.Module:
    """Execute the API transformation pass over ``module``.

    Args:
        module: Source CST module.
        context: Shared ``RewriterContext``.

    Returns:
        Transformed CST module.
    """
    transformer = ApiTransformer(context)
    return module.visit(transformer)


class ApiTransformer(cst.CSTTransformer):
  """Single-inheritance LibCST transformer composing ``ApiPassState`` and rewrite handlers.

  Attributes:
      context: Shared ``RewriterContext`` for the current pass.
      state: Typed ``ApiPassState`` holding trait caches and scope state helpers.
      resolver: ``MappingResolver`` handling FQN resolution, variant lookups, and injections.
      attr_handler: ``AttributeRewriteHandler`` handling ``Assign`` and ``Attribute`` nodes.
      call_handler: ``CallRewriteHandler`` handling ``Call`` expressions.
  """

  def __init__(self, context: RewriterContext) -> None:
    """Initialize ``ApiTransformer`` and its composed handlers over ``context``.

    Args:
        context: Shared ``RewriterContext``.
    """
    self.context: RewriterContext = context
    self.state: ApiPassState = ApiPassState(context=context)
    self.resolver: MappingResolver = MappingResolver(self.state)
    self.attr_handler: AttributeRewriteHandler = AttributeRewriteHandler(self)
    self.call_handler: CallRewriteHandler = CallRewriteHandler(self)

  @property
  def _cached_source_traits(self) -> StructuralTraits | None:
    """Return cached source ``StructuralTraits`` stored on ``self.state``.

    Returns:
        Cached source ``StructuralTraits`` or ``None``.
    """
    return self.state.cached_source_traits

  @_cached_source_traits.setter
  def _cached_source_traits(self, value: StructuralTraits | None) -> None:
    """Update cached source ``StructuralTraits`` stored on ``self.state``.

    Args:
        value: ``StructuralTraits`` instance or ``None``.
    """
    self.state.cached_source_traits = value

  @property
  def _cached_target_traits(self) -> StructuralTraits | None:
    """Return cached target ``StructuralTraits`` stored on ``self.state``.

    Returns:
        Cached target ``StructuralTraits`` or ``None``.
    """
    return self.state.cached_target_traits

  @_cached_target_traits.setter
  def _cached_target_traits(self, value: StructuralTraits | None) -> None:
    """Update cached target ``StructuralTraits`` stored on ``self.state``.

    Args:
        value: ``StructuralTraits`` instance or ``None``.
    """
    self.state.cached_target_traits = value

  @property
  def semantics(self) -> "SemanticsManager":
    """Return the ``SemanticsManager`` bound to ``self.state``.

    Returns:
        Active ``SemanticsManager``.
    """
    return self.state.semantics

  @property
  def config(self) -> RuntimeConfig:
    """Return the ``RuntimeConfig`` bound to ``self.state``.

    Returns:
        Active ``RuntimeConfig``.
    """
    return self.state.config

  @property
  def source_framework(self) -> str:
    """Return the source framework identifier.

    Returns:
        Source framework name.
    """
    return self.state.source_framework

  @property
  def target_framework(self) -> str:
    """Return the target framework identifier.

    Returns:
        Target framework name.
    """
    return self.state.target_framework

  @property
  def strict_mode(self) -> bool:
    """Return whether strict mode is enabled.

    Returns:
        ``True`` if strict mode is active.
    """
    return self.state.strict_mode

  @property
  def ctx(self) -> HookContext:
    """Return the ``HookContext`` exposed for plugin execution.

    Returns:
        Bound ``HookContext``.
    """
    return self.state.ctx

  @property
  def source_traits(self) -> StructuralTraits:
    """Return lazily loaded ``StructuralTraits`` for the source framework.

    Returns:
        Source ``StructuralTraits``.
    """
    return self.state.source_traits

  @property
  def target_traits(self) -> StructuralTraits:
    """Return lazily loaded ``StructuralTraits`` for the target framework.

    Returns:
        Target ``StructuralTraits``.
    """
    return self.state.target_traits

  def _get_target_traits(self) -> StructuralTraits:
    """Return lazily loaded ``StructuralTraits`` for the target framework.

    Returns:
        Target ``StructuralTraits``.
    """
    return self.state.target_traits

  def _get_source_lifecycle_lists(self) -> tuple[set[str], set[str]]:
    """Return strip and warning lifecycle method sets for the source framework.

    Returns:
        Tuple of ``(strip_methods, warn_methods)``.
    """
    return self.state.get_source_lifecycle_lists()

  def _report_failure(self, reason: str) -> None:
    """Record a statement-level failure message.

    Args:
        reason: Failure description string.
    """
    self.state.report_failure(reason)

  def _report_warning(self, reason: str) -> None:
    """Record a statement-level warning message.

    Args:
        reason: Warning description string.
    """
    self.state.report_warning(reason)

  def _mark_stateful(self, var_name: str) -> None:
    """Mark ``var_name`` as stateful in the innermost lexical scope.

    Args:
        var_name: Variable name to mark stateful.
    """
    self.state.mark_stateful(var_name)

  def _is_stateful(self, var_name: str) -> bool:
    """Return whether ``var_name`` is marked stateful in any active scope.

    Args:
        var_name: Variable name to inspect.

    Returns:
        ``True`` if ``var_name`` is stateful.
    """
    return self.state.is_stateful(var_name)

  # --- Delegated Resolver Methods (supporting RewriterHost protocol) ---

  def _cst_to_string(self, node: cst.CSTNode) -> str | None:
    """Delegate CST expression flattening to ``self.resolver``.

    Args:
        node: CST node to flatten.

    Returns:
        Dotted string representation or ``None``.
    """
    return self.resolver.cst_to_string(node)

  def _get_qualified_name(self, node: cst.BaseExpression) -> str | None:
    """Delegate alias-resolved FQN lookup to ``self.resolver``.

    Args:
        node: CST expression node.

    Returns:
        Fully qualified name string or ``None``.
    """
    return self.resolver.get_qualified_name(node, cst_to_string_fn=self._cst_to_string)

  def _create_name_node(self, api_path: str) -> cst.Name | cst.Attribute:
    """Delegate CST ``Name``/``Attribute`` construction to ``self.resolver``.

    Args:
        api_path: Dotted API path string.

    Returns:
        Constructed ``cst.Name`` or ``cst.Attribute``.
    """
    return self.resolver.create_name_node(api_path)

  def _create_dotted_name(self, name_str: str) -> cst.Name | cst.Attribute:
    """Delegate dotted CST expression construction to ``self.resolver``.

    Args:
        name_str: Dotted name string.

    Returns:
        Constructed ``cst.Name`` or ``cst.Attribute``.
    """
    return self.resolver.create_dotted_name(name_str)

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """Delegate module-alias detection to ``self.resolver``.

    Args:
        node: CST node to inspect.

    Returns:
        ``True`` if ``node`` references a module alias or framework root.
    """
    return self.resolver.is_module_alias(node, cst_to_string_fn=self._cst_to_string)

  def _apply_preamble(self, node: cst.FunctionDef, stmts_code: list[str]) -> cst.FunctionDef:
    """Delegate function preamble statement injection to ``self.resolver``.

    Args:
        node: Target ``cst.FunctionDef`` node.
        stmts_code: List of Python statement strings to inject.

    Returns:
        Updated ``cst.FunctionDef``.
    """
    return self.resolver.apply_preamble(node, stmts_code)

  def _inject_stmts_to_body(
    self,
    node: cst.FunctionDef,
    new_stmts: typing.Sequence[cst.SimpleStatementLine | cst.BaseCompoundStatement],
  ) -> cst.FunctionDef:
    """Delegate statement body insertion to ``self.resolver``.

    Args:
        node: Target ``cst.FunctionDef`` node.
        new_stmts: Statements to insert after any docstring.

    Returns:
        Updated ``cst.FunctionDef``.
    """
    return self.resolver.inject_stmts_to_body(node, new_stmts)

  def _convert_to_indented_block(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """Delegate one-liner suite normalization to ``self.resolver``.

    Args:
        node: ``cst.FunctionDef`` node to normalize.

    Returns:
        Updated ``cst.FunctionDef``.
    """
    return self.resolver.convert_to_indented_block(node)

  def _get_mapping(self, name: str, silent: bool = False) -> dict[str, JsonValue] | None:
    """Delegate semantic variant lookup to ``self.resolver``.

    Args:
        name: Fully qualified source API name.
        silent: Whether to suppress failure reporting.

    Returns:
        Target variant mapping dict or ``None``.
    """
    return self.resolver.get_mapping(name, silent=silent, report_failure_fn=self._report_failure)

  def _handle_variant_imports(self, variant: typing.Mapping[str, RuntimeTensorOrValue]) -> None:
    """Delegate variant ``required_imports`` preamble injection to ``self.resolver``.

    Args:
        variant: Target variant mapping dictionary.
    """
    self.resolver.handle_variant_imports(variant)

  def _is_framework_base(self, name: str) -> bool:
    """Delegate framework module-base detection to ``self.resolver``.

    Args:
        name: Candidate class name.

    Returns:
        ``True`` if ``name`` matches a known framework module base.
    """
    return self.resolver.is_framework_base(name)

  def check_version_constraints(self, min_v: str | None, max_v: str | None) -> str | None:
    """Delegate version constraint checking to ``self.resolver``.

    Args:
        min_v: Optional minimum version string.
        max_v: Optional maximum version string.

    Returns:
        Warning string if violated, else ``None``.
    """
    return self.resolver.check_version_constraints(min_v, max_v)

  def _inject_argument_to_signature(
    self,
    node: cst.FunctionDef,
    arg_name: str,
    annotation: str | None,
  ) -> cst.FunctionDef:
    """Delegate function parameter injection to ``self.resolver``.

    Args:
        node: Target ``cst.FunctionDef``.
        arg_name: Parameter name to inject.
        annotation: Optional type annotation string.

    Returns:
        Updated ``cst.FunctionDef``.
    """
    return self.resolver.inject_argument_to_signature(node, arg_name, annotation)

  # --- CST Visitor & Transformer Hooks ---

  def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
    """Delegate ``cst.Assign`` rewriting to ``self.attr_handler``.

    Args:
        original_node: Assign node prior to child traversal.
        updated_node: Assign node after child traversal.

    Returns:
        Transformed or unchanged ``cst.Assign``.
    """
    return self.attr_handler.rewrite_assign(original_node, updated_node)

  def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
    """Delegate ``cst.Attribute`` rewriting to ``self.attr_handler``.

    Args:
        original_node: Attribute node prior to child traversal.
        updated_node: Attribute node after child traversal.

    Returns:
        Transformed ``cst.BaseExpression``.
    """
    return self.attr_handler.rewrite_attribute(original_node, updated_node)

  def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
    """Delegate ``cst.Call`` rewriting to ``self.call_handler``.

    Args:
        original_node: Call node prior to child traversal.
        updated_node: Call node after child traversal.

    Returns:
        Transformed ``cst.BaseExpression``.
    """
    return self.call_handler.rewrite_call(original_node, updated_node)

  def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
    """Flush accumulated module-level preamble statements onto ``updated_node``.

    Args:
        original_node: Original ``cst.Module`` prior to traversal.
        updated_node: Transformed ``cst.Module``.

    Returns:
        Updated ``cst.Module`` with deduplicated preamble statements prepended.
    """
    if not self.context.module_preamble:
      return updated_node

    new_stmts: list[cst.SimpleStatementLine | cst.BaseCompoundStatement] = []
    seen: set[str] = set()
    for code in self.context.module_preamble:
      if code in seen:
        continue
      seen.add(code)
      try:
        mod = cst.parse_module(code)
        new_stmts.extend(mod.body)
      except cst.ParserSyntaxError:
        pass

    self.context.module_preamble.clear()

    if not new_stmts:
      return updated_node

    return updated_node.with_changes(body=new_stmts + list(updated_node.body))

  def visit_ClassDef(self, node: cst.ClassDef) -> bool | None:
    """Enter class lexical scope and detect framework ``Module`` subclasses.

    Args:
        node: ``cst.ClassDef`` being visited.

    Returns:
        ``True`` to traverse child nodes.
    """
    self.context.scope_stack.append(set())

    is_module = False
    for base in node.bases:
      name = self._get_qualified_name(base.value)
      if name and self._is_framework_base(name):
        is_module = True
        break

    if not is_module:
      for base in node.bases:
        raw_name = self._cst_to_string(base.value)
        if raw_name and self._is_framework_base(raw_name):
          is_module = True
          break

    if is_module:
      self.context.in_module_class = True

    return True

  def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.ClassDef:
    """Pop class lexical scope and reset ``in_module_class``.

    Args:
        original_node: ``cst.ClassDef`` prior to child traversal.
        updated_node: ``cst.ClassDef`` after child traversal.

    Returns:
        Unchanged ``updated_node``.
    """
    self.context.scope_stack.pop()
    if self.context.in_module_class:
      self.context.in_module_class = False
    return updated_node

  def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
    """Push function lexical scope and ``SignatureContext`` frame.

    Args:
        node: ``cst.FunctionDef`` being visited.

    Returns:
        ``True`` to traverse child nodes.
    """
    self.context.scope_stack.append(set())

    existing_args = {param.name.value for param in node.params.params}
    is_init = node.name.value == "__init__"
    self.context.signature_stack.append(
      SignatureContext(
        existing_args=existing_args,
        is_init=is_init,
        is_module_method=self.context.in_module_class,
      )
    )
    return True

  def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
    """Pop function scope and apply pending signature argument and preamble injections.

    Args:
        original_node: ``cst.FunctionDef`` prior to child traversal.
        updated_node: ``cst.FunctionDef`` after child traversal.

    Returns:
        Updated ``cst.FunctionDef`` with injected parameters and preamble statements.
    """
    self.context.scope_stack.pop()

    if self.context.signature_stack:
      sig_ctx = self.context.signature_stack.pop()
      for name, annotation in sig_ctx.injected_args:
        updated_node = self._inject_argument_to_signature(updated_node, name, annotation)
      if sig_ctx.preamble_stmts:
        updated_node = self._apply_preamble(updated_node, sig_ctx.preamble_stmts)

    return updated_node

  def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> bool | None:
    """Reset per-statement error and warning buffers.

    Args:
        node: ``cst.SimpleStatementLine`` being visited.

    Returns:
        ``True`` to traverse child nodes.
    """
    self.context.current_stmt_errors = []
    self.context.current_stmt_warnings = []
    return True

  def leave_SimpleStatementLine(
    self,
    original_node: cst.SimpleStatementLine,
    updated_node: cst.SimpleStatementLine,
  ) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement] | cst.RemovalSentinel:
    """Wrap ``original_node`` or ``updated_node`` via ``EscapeHatch`` if diagnostics were recorded.

    Args:
        original_node: ``cst.SimpleStatementLine`` prior to child traversal.
        updated_node: ``cst.SimpleStatementLine`` after child traversal.

    Returns:
        Updated statement or ``EscapeHatch``-marked statement sentinel.
    """
    if self.context.current_stmt_errors:
      unique_errors = list(dict.fromkeys(self.context.current_stmt_errors))
      return EscapeHatch.mark_failure(original_node, "; ".join(unique_errors))

    if self.context.current_stmt_warnings:
      unique_warnings = list(dict.fromkeys(self.context.current_stmt_warnings))
      return EscapeHatch.mark_failure(updated_node, "; ".join(unique_warnings))

    return updated_node

  def visit_Import(self, node: cst.Import) -> bool | None:
    """Record ``import`` aliases in ``self.context.alias_map``.

    Args:
        node: ``cst.Import`` node being visited.

    Returns:
        ``False`` because import children do not require expression rewriting.
    """
    for alias in node.names:
      full_name = self._cst_to_string(alias.name)
      if not full_name:
        continue

      if alias.asname and isinstance(alias.asname.name, cst.Name):
        self.context.alias_map[alias.asname.name.value] = full_name
      else:
        root = full_name.split(".")[0]
        self.context.alias_map[root] = root
    return False

  def visit_ImportFrom(self, node: cst.ImportFrom) -> bool | None:
    """Record ``from ... import ...`` aliases in ``self.context.alias_map``.

    Args:
        node: ``cst.ImportFrom`` node being visited.

    Returns:
        ``False`` because import children do not require expression rewriting.
    """
    if node.relative:
      return False

    module_name = self._cst_to_string(node.module) if node.module else ""
    if not module_name or isinstance(node.names, cst.ImportStar):
      return False

    for alias in node.names:
      name_obj = alias.name
      imported_name = name_obj.value if isinstance(name_obj, cst.Name) else name_obj.attr.value
      canonical_source = f"{module_name}.{imported_name}"
      local_name = alias.asname.name.value if alias.asname and isinstance(alias.asname.name, cst.Name) else imported_name
      self.context.alias_map[local_name] = canonical_source

    return False


__all__ = [
  "ApiPass",
  "ApiPassState",
  "ApiTransformer",
  "AttributeRewriteHandler",
  "CallRewriteHandler",
  "MappingResolver",
]
