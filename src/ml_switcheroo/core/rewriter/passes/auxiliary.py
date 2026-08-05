"""Auxiliary Logic Pass.

This module consolidates transformations that are adjacent to the core API logic,
specifically:

 **Decorators**: Rewriting or stripping function decorators (e.g., `@torch.jit.script`).

**Control Flow**: Analyzing and transforming loops (`for`, `while`) to ensure safety
    in functional frameworks (e.g., JAX/XLA compatibility checks).

It merges logic previously found in `decorators.py` and `control_flow.py`.
"""

from typing import Any

from typing import Union, Optional
import libcst as cst

from ml_switcheroo.core.rewriter.interface import RewriterPass
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.hooks import get_hook
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.semantics.schema import StructuralTraits


class AuxiliaryPass(RewriterPass):
  """Pass dealing with auxiliary syntax constructs: decorators and control flow."""

  def transform(self, module: cst.Module, context: RewriterContext) -> cst.Module:
    """Executes the auxiliary transformation logic.

    Args:
        module: The source CST.
        context: Shared state.

    Returns:
        The transformed CST.
    """
    transformer = AuxiliaryTransformer(context)
    return module.visit(transformer)


class AuxiliaryTransformer(cst.CSTTransformer):
  """LibCST Transformer for auxiliary constructs.

  Attributes:
      context: The shared execution context.
  """

  def __init__(self, context: RewriterContext) -> None:
    """Initialize.

    Args:
        context: The execution context.
    """
    self.context = context
    self._cached_traits: Optional[StructuralTraits] = None

  # --- Properties ---

  def _get_traits(self) -> StructuralTraits:
    """Lazily loads target structural traits.

    Returns:
        The StructuralTraits instance for the target framework.
    """
    if self._cached_traits:
      return self._cached_traits

    conf = self.context.semantics.get_framework_config(self.context.target_fw)
    if conf and "traits" in conf:
      self._cached_traits = StructuralTraits.model_validate(conf["traits"])
    else:
      self._cached_traits = StructuralTraits()

    return self._cached_traits

  def _get_qualified_name(self, node: cst.BaseExpression) -> Optional[str]:
    """Resolves node to string using context alias map.

    Args:
        node: The CST expression node to resolve.

    Returns:
        The canonical qualified name as a dot-separated string, or None if the
        expression cannot be resolved.
    """
    full_str = self._cst_to_string(node)
    if not full_str:
      return None

    parts = full_str.split(".")
    root = parts[0]

    if root in self.context.alias_map:
      canonical_root = self.context.alias_map[root]
      if len(parts) > 1:
        return f"{canonical_root}.{'.'.join(parts[1:])}"
      return canonical_root

    return full_str

  def _cst_to_string(self, node: cst.BaseExpression) -> Optional[str]:
    """Flatten CST to dot-string.

    Args:
        node: The CST expression node to flatten.

    Returns:
        The flattened dot-separated string representation of the node, or None
        if the node type is not supported.
    """
    if isinstance(node, cst.Name):
      return node.value
    elif isinstance(node, cst.Attribute):
      base = self._cst_to_string(node.value)
      if base:
        return f"{base}.{node.attr.value}"
    return None

  def _create_dotted_name(self, name_str: str) -> cst.BaseExpression:
    """Creates CST node from string.

    Args:
        name_str: A dot-separated name string.

    Returns:
        The constructed CST expression node representing the dotted name.
    """
    parts = name_str.split(".")
    node = cst.Name(parts[0])
    for part in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(part))  # type: ignore
    return node

  # --- Error Handling ---

  def _report_failure(self, reason: str) -> None:
    """Report error to context.

    Args:
        reason: The error message string describing the failure.
    """
    self.context.current_stmt_errors.append(reason)

  def _report_warning(self, reason: str) -> None:
    """Report warning to context.

    Args:
        reason: The warning message string describing the issue.
    """
    self.context.current_stmt_warnings.append(reason)

  def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> Optional[bool]:
    """Reset statement buffers.

    Args:
        node: The SimpleStatementLine node being visited.

    Returns:
        True to continue visiting children, or None to stop.
    """
    self.context.current_stmt_errors = []
    self.context.current_stmt_warnings = []
    return True

  def leave_SimpleStatementLine(
    self,
    original_node: cst.SimpleStatementLine,
    updated_node: cst.SimpleStatementLine,
  ) -> Union[cst.SimpleStatementLine, cst.FlattenSentinel[Any]]:
    """Process statement errors.

    Args:
        original_node: The original SimpleStatementLine node.
        updated_node: The updated SimpleStatementLine node.

    Returns:
        The updated node, or an error-marked node if warnings/errors occurred.
    """
    # Check warnings
    if self.context.current_stmt_warnings:
      unique = list(dict.fromkeys(self.context.current_stmt_warnings))
      msg = "; ".join(unique)
      return EscapeHatch.mark_failure(updated_node, msg)  # type: ignore

    # Check errors (Priority over warnings for reversion logic structure)
    if self.context.current_stmt_errors:
      unique = list(dict.fromkeys(self.context.current_stmt_errors))
      msg = "; ".join(unique)
      return EscapeHatch.mark_failure(original_node, msg)  # type: ignore

    return updated_node

  # --- Decorator Logic ---

  def leave_Decorator(
    self, original_node: cst.Decorator, updated_node: cst.Decorator
  ) -> Union[cst.Decorator, cst.RemovalSentinel]:
    """Rewrites decorators.

    Args:
        original_node: The original Decorator node.
        updated_node: The updated Decorator node.

    Returns:
        The renamed decorator, a removal sentinel, or the unchanged node.
    """
    # Extract expression
    expr = original_node.decorator
    func_node = expr.func if isinstance(expr, cst.Call) else expr

    name = self._get_qualified_name(func_node)
    if not name:
      return updated_node

    lookup = self.context.semantics.get_definition(name)
    if not lookup:
      return updated_node

    _, details = lookup
    variants = details.get("variants", {})

    if self.context.target_fw not in variants:
      return updated_node

    target_variant = variants[self.context.target_fw]

    # Case: Removal
    if target_variant is None:
      return cst.RemoveFromParent()

    # Case: Rename
    target_api = target_variant.get("api")
    if target_api:
      new_name_node = self._create_dotted_name(target_api)
      current_expr = updated_node.decorator

      if isinstance(current_expr, cst.Call):
        new_expr = current_expr.with_changes(func=new_name_node)
      else:
        new_expr = new_name_node  # type: ignore

      return updated_node.with_changes(decorator=new_expr)

    return updated_node

  # --- Control Flow Logic ---

  def leave_For(self, original_node: cst.For, updated_node: cst.For) -> Union[cst.For, cst.CSTNode, cst.FlattenSentinel]:  # type: ignore
    """Processes 'for' loops for safety checks and unrolling.

    Args:
        original_node: The original For loop node.
        updated_node: The updated For loop node.

    Returns:
        The transformed loop node, unrolled statement list, or the original node in case of errors.
    """
    # 1. Attempt Static Unroll (Optimization Hook)
    static_hook = get_hook("transform_for_loop_static")
    if static_hook:
      try:
        new_node = static_hook(updated_node, self.context.hook_context)
        if new_node is not updated_node:
          return new_node
      except Exception as e:
        import traceback

        traceback.print_exc()
        self._report_warning(f"Static loop unrolling failed: {str(e)}")

    # -  General Safety Check Hook
    hook = get_hook("transform_for_loop")
    if hook:
      try:
        new_node = hook(updated_node, self.context.hook_context)
        if new_node is not updated_node:
          return new_node
      except Exception as e:
        self._report_failure(f"Loop transformation failed: {str(e)}")
        # Since Compound statements aren't handled by leave_SimpleStatementLine error logic,
        # we must wrap manually here.
        return EscapeHatch.mark_failure(original_node, f"Loop transformation failed: {str(e)}")

    return updated_node
