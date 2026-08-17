"""API Logic Pass.

This module consolidates all API-level transformations, including:

1.  **Function Calls**: Remapping APIs (e.g., `torch.abs` -> `jnp.abs`), applying argument normalization,
    handling layout permutations, and executing transformation strategies (infix, lambda, macros).
2.  **Attributes**: Remapping attributes/constants (e.g., `torch.float32` -> `jnp.float32`).
3.  **Assignments**: Unwrapping functional return patterns (e.g., `layer.apply`).
4.  **Symbol Resolution**: Resolving aliases to fully qualified names for lookup.
5.  **Scoping**: Tracking stateful variables (layers) to inform call rewriting logic.
"""

from __future__ import annotations
from typing import Any

from typing import Optional, Set, Tuple, Union
import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.interface import RewriterPass
from ml_switcheroo.core.rewriter.types import SignatureContext
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits

from ml_switcheroo.core.rewriter.passes.api_helpers import ApiHelpersMixin
from ml_switcheroo.core.rewriter.passes.api_attr_mixin import ApiTransformerAttrMixin
from ml_switcheroo.core.rewriter.passes.api_call_mixin import ApiTransformerCallMixin


class ApiPass(RewriterPass):
  """Transformation pass for rewiring API usage.

  Handles resolving function calls to Abstract Operations (The Hub) and then projecting
  them to the Target Framework (The Spoke). Also handles attribute renaming and
  stateful assignment tracking.
  """

  def transform(self, module: cst.Module, context: RewriterContext) -> cst.Module:
    """Executes the API transformation logic.

    Args:
        module: The source CST.
        context: Shared rewriter state.

    Returns:
        The transformed CST.
    """
    transformer = ApiTransformer(context)
    return module.visit(transformer)


class ApiTransformer(ApiHelpersMixin, ApiTransformerAttrMixin, ApiTransformerCallMixin, cst.CSTTransformer):
  """LibCST Transformer for API Logic.

  This class centralizes the logic for:
  - Resolving names/aliases.
  - Tracking scope/state.
  - Rewriting Calls, Attributes, and Assignments.
  """

  def __init__(self, context: RewriterContext) -> None:
    """Initialize the transformer.

    Args:
        context: The shared rewriter context.
    """
    self.context = context
    self._cached_source_traits: Optional[StructuralTraits] = None
    self._cached_target_traits: Optional[StructuralTraits] = None

  # --- Properties & Helpers ---

  @property
  def semantics(self) -> SemanticsManager:
    """Accessor for semantics manager.

    Returns:
        The semantics manager instance bound to the current context.
    """
    return self.context.semantics

  @property
  def config(self) -> RuntimeConfig:
    """Accessor for runtime config.

    Returns:
        The runtime configuration instance bound to the current context.
    """
    return self.context.config

  @property
  def source_fw(self) -> str:
    """Accessor for source framework key.

    Returns:
        The string identifier for the source framework (e.g., 'torch').
    """
    return self.context.source_fw

  @property
  def target_fw(self) -> str:
    """Accessor for target framework key.

    Returns:
        The string identifier for the target framework (e.g., 'jax').
    """
    return self.context.target_fw

  @property
  def strict_mode(self) -> bool:
    """Accessor for strict mode flag.

    Returns:
        True if strict mode is enabled, False otherwise.
    """
    return self.config.strict_mode

  @property
  def source_traits(self) -> StructuralTraits:
    """Lazily loads source framework traits.

    Returns:
        The structural traits configuration of the source framework.
    """
    if self._cached_source_traits:
      return self._cached_source_traits

    config_dict = self.semantics.get_framework_config(self.source_fw)
    if config_dict and "traits" in config_dict:
      self._cached_source_traits = StructuralTraits.model_validate(config_dict["traits"])
    else:
      self._cached_source_traits = StructuralTraits()
    return self._cached_source_traits

  def _get_target_traits(self) -> StructuralTraits:
    """Lazily loads target framework traits.

    Returns:
        The structural traits configuration of the target framework.
    """
    if self._cached_target_traits:
      return self._cached_target_traits

    config_dict = self.semantics.get_framework_config(self.target_fw)
    if config_dict and "traits" in config_dict:
      self._cached_target_traits = StructuralTraits.model_validate(config_dict["traits"])
    else:
      self._cached_target_traits = StructuralTraits()

    return self._cached_target_traits

  def _get_source_lifecycle_lists(self) -> Tuple[Set[str], Set[str]]:
    """Returns strip and warn method sets for lifecycle management.

    Returns:
        A tuple containing two sets:
        - The first set contains method names that should be stripped.
        - The second set contains method names that should trigger a warning.
    """
    traits = self.source_traits
    return (
      set(traits.lifecycle_strip_methods),
      set(traits.lifecycle_warn_methods),
    )

  def _report_failure(self, reason: str) -> None:
    """Records a failure in the context error buffer.

    Args:
        reason: A message describing the cause of the failure.
    """
    self.context.current_stmt_errors.append(reason)

  def _report_warning(self, reason: str) -> None:
    """Records a warning in the context warning buffer.

    Args:
        reason: A message describing the warning condition.
    """
    self.context.current_stmt_warnings.append(reason)

  # --- Preamble Output ---
  # Use simple implementation for now without deduplication on identity which might be complex,
  # as simple set of strings works for identical injects.
  def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
    """Injects accumulated module-level preamble statements if they haven't been flushed yet.

    If preambles were gathered during traversal, inject them into the module's body.
    We deduplicate based on string content.

    Args:
        original_node: The CST node representing the original module before traversal.
        updated_node: The transformed CST module node.

    Returns:
        The final transformed CST module containing any newly injected preamble statements.
    """
    if not self.context.module_preamble:
      return updated_node

    new_stmts = []  # type: ignore
    seen = set()
    for code in self.context.module_preamble:
      if code in seen:
        continue
      seen.add(code)
      try:
        mod = cst.parse_module(code)
        new_stmts.extend(mod.body)
      except Exception:
        pass

    # Clear buffer to prevent re-injection
    self.context.module_preamble.clear()

    if not new_stmts:
      return updated_node

    return updated_node.with_changes(body=new_stmts + list(updated_node.body))

  # --- Scoping Logic ---

  def _mark_stateful(self, var_name: str) -> None:
    """Marks variable as stateful in current scope.

    Args:
        var_name: The name of the variable being assigned a stateful component.
    """
    if self.context.scope_stack:  # pragma: no branch
      self.context.scope_stack[-1].add(var_name)

  def _is_stateful(self, var_name: str) -> bool:
    """Checks if variable is stateful (traversing up scopes).

    Args:
        var_name: The name of the variable to check.

    Returns:
        True if the variable is marked stateful in the current or any enclosing scope.
    """
    for scope in reversed(self.context.scope_stack):
      if var_name in scope:
        return True
    return False

  # --- Context Stack Mirroring (Essential for Preamble Injection in Logic Plugins) ---

  def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
    """Enter class scope and detect Module.

    Args:
        node: The CST ClassDef node being visited.

    Returns:
        True to continue traversing children.
    """
    self.context.scope_stack.append(set())

    is_module = False
    for base in node.bases:
      name = self._get_qualified_name(base.value)
      if name and self._is_framework_base(name):
        is_module = True
        break

    # Fallback raw check
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
    """Exit class scope.

    Args:
        original_node: The CST ClassDef node prior to visiting children.
        updated_node: The transformed CST ClassDef node after visiting children.

    Returns:
        The updated CST ClassDef node.
    """
    self.context.scope_stack.pop()
    if self.context.in_module_class:
      self.context.in_module_class = False
    return updated_node

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    """Enter function scope.

    Args:
        node: The CST FunctionDef node being visited.

    Returns:
        True to continue traversing children.
    """
    self.context.scope_stack.append(set())

    existing_args = set()
    for param in node.params.params:
      if isinstance(param.name, cst.Name):  # pragma: no branch
        existing_args.add(param.name.value)

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
    """Exit function scope.

    Flush any pending preamble statements requested by plugins during this pass.
    Also apply any pending signature injections (arguments).

    Args:
        original_node: The CST FunctionDef node prior to visiting children.
        updated_node: The transformed CST FunctionDef node.

    Returns:
        The updated CST FunctionDef node with any injected arguments or preamble statements.
    """
    self.context.scope_stack.pop()

    if self.context.signature_stack:
      sig_ctx = self.context.signature_stack.pop()

      # 1. Apply Argument Injection (e.g. from Plugins like rng_threading)
      # New Logic: ApiPass can now modify signatures if plugins request it
      for name, annotation in sig_ctx.injected_args:
        updated_node = self._inject_argument_to_signature(updated_node, name, annotation)

      # 2. Apply Preambles
      if sig_ctx.preamble_stmts:
        updated_node = self._apply_preamble(updated_node, sig_ctx.preamble_stmts)

    return updated_node

  # --- Error Handling & Statement Processing ---

  def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> Optional[bool]:
    """Reset statement-level error buffers.

    Args:
        node: The CST SimpleStatementLine node being visited.

    Returns:
        True to continue traversing children.
    """
    self.context.current_stmt_errors = []
    self.context.current_stmt_warnings = []
    return True

  def leave_SimpleStatementLine(
    self,
    original_node: cst.SimpleStatementLine,
    updated_node: cst.SimpleStatementLine,
  ) -> Union[cst.SimpleStatementLine, cst.FlattenSentinel[Any]]:
    """Check for errors generated by child expressions and wrap if needed.

    Args:
        original_node: The CST node prior to transformation.
        updated_node: The transformed CST node.

    Returns:
        The transformed line, possibly wrapped via EscapeHatch if errors or warnings were logged.
    """
    if self.context.current_stmt_errors:
      unique_errors = list(dict.fromkeys(self.context.current_stmt_errors))
      message = "; ".join(unique_errors)
      return EscapeHatch.mark_failure(original_node, message)  # type: ignore

    if self.context.current_stmt_warnings:
      unique_warnings = list(dict.fromkeys(self.context.current_stmt_warnings))
      message = "; ".join(unique_warnings)
      return EscapeHatch.mark_failure(updated_node, message)  # type: ignore

    return updated_node

  # --- Resolver Logic ---

  def visit_Import(self, node: cst.Import) -> Optional[bool]:
    """Track import aliases.

    Args:
        node: The CST Import node being visited.

    Returns:
        False to prevent deeper traversal, since aliases are fully captured here.
    """
    for alias in node.names:
      full_name = self._cst_to_string(alias.name)
      if not full_name:
        continue

      if alias.asname:
        local_name = alias.asname.name.value  # type: ignore
        self.context.alias_map[local_name] = full_name
      else:
        root = full_name.split(".")[0]
        self.context.alias_map[root] = root
    return False

  def visit_ImportFrom(self, node: cst.ImportFrom) -> Optional[bool]:
    """Track from-import aliases.

    Args:
        node: The CST ImportFrom node being visited.

    Returns:
        False to prevent deeper traversal.
    """
    if node.relative:
      return False

    module_name = self._cst_to_string(node.module) if node.module else ""
    if not module_name:
      return False

    if isinstance(node.names, cst.ImportStar):
      return False

    for alias in node.names:
      if not isinstance(alias, cst.ImportAlias):
        continue
      imported_name = alias.name.value
      canonical_source = f"{module_name}.{imported_name}"
      local_name = alias.asname.name.value if alias.asname else imported_name  # type: ignore
      self.context.alias_map[local_name] = canonical_source  # type: ignore

    return False

  # --- Hook Accessors (Proxy) ---
  @property
  def ctx(self) -> Any:
    """Expose hook context for strategy invocation.

    Returns:
        The hook context associated with this traversal.
    """
    return self.context.hook_context
