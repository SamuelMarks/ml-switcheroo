"""
Base Rewriter Implementation with Alias and Scope Resolution.

This module provides the ``BaseRewriter`` class, which serves as the foundation for
the ``PivotRewriter``. It handles:

1.  **State Management**: Tracking the current scope (global vs class vs function)
    to handle stateful variable detection.
2.  **Alias Resolution**: Tracking ``import as`` statements to resolve ``t.abs`` back
    to ``torch.abs`` or ``np.sum`` to ``numpy.sum``.
3.  **Error Reporting**: Collecting failures during the AST walk to be bubbled
    up to the ``ASTEngine``.
4.  **Hook Infrastructure**: initializing the ``HookContext`` used by plugins.
5.  **Global Injection**: Handling file-level preamble injection (``leave_Module``).
6.  **Import Injection**: Processing dynamic import requirements from variants.
"""

from typing import Optional, List, Dict, Any, Union, Set
import libcst as cst

from ml_switcheroo.core.tracer import get_tracer
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.types import SignatureContext


class BaseRewriter(cst.CSTTransformer):
  """
  The base class for AST transformation traversal.

  Provides common utilities for scope tracking, alias resolution, and error bubbling,
  which are utilized by the specific Mixins (CallMixin, StructureMixin, etc.).
  """

  def __init__(self, semantics: SemanticsManager, config: RuntimeConfig):
    """
    Initializes the rewriter.

    Args:
        semantics: The SemanticsManager instance containing the knowledge graph.
        config: The runtime configuration object.
    """
    self.semantics = semantics
    self.config = config

    # Directly use strings from config properties
    # FIX: Use effective properties to respecting Flavour overrides (e.g. flax_nnx)
    self.source_fw = str(config.effective_source)
    self.target_fw = str(config.effective_target)

    self.strict_mode = config.strict_mode

    self.ctx = HookContext(
      semantics,
      config,
      arg_injector=self._callback_inject_arg,
      preamble_injector=self._callback_inject_preamble,
    )
    self._current_stmt_errors: List[str] = []
    self._current_stmt_warnings: List[str] = []

    # Stack of scopes. Each scope is a set of variable names considered "stateful".
    # Index -1 is the current scope.
    self._scope_stack: List[Set[str]] = [set()]
    self._signature_stack: List[SignatureContext] = []
    self._in_module_class = False
    self._alias_map: Dict[str, str] = {}

    # Store for global injections
    self._module_preamble: List[str] = []

  def _callback_inject_arg(self, name: str, annotation: Optional[str] = None) -> None:
    """
    Callback for plugins to inject arguments into the current function signature.

    Args:
        name: The argument name (e.g. 'rng').
        annotation: Optional type hint string.
    """
    if not self._signature_stack:
      return

    ctx = self._signature_stack[-1]
    if name not in ctx.existing_args:
      found = any(existing_name == name for existing_name, _ in ctx.injected_args)
      if not found:
        ctx.injected_args.append((name, annotation))

  def _callback_inject_preamble(self, code_str: str) -> None:
    """
    Callback for plugins to inject statements.
    If inside a function, injects at the start of the function body.
    If at module level (no active function stack), injects at the top of the file.

    Args:
        code_str: The code to inject (e.g. 'import foo', 'class Shim...').
    """
    if not self._signature_stack:
      # Module level injection
      if code_str not in self._module_preamble:
        self._module_preamble.append(code_str)
      return

    # Function level injection
    ctx = self._signature_stack[-1]
    if code_str not in ctx.preamble_stmts:
      ctx.preamble_stmts.append(code_str)

  def _handle_variant_imports(self, variant: Dict[str, Any]) -> None:
    """
    Processes `required_imports` from a variant specification.
    Injects valid import statements into the module preamble via the Context.

    Supports:
    - List of strings: `["import numpy"]`
    - List of structured specifications: `[{"module": "numpy", "alias": "np"}]`

    Args:
        variant: The framework variant dictionary from Semantics.
    """
    reqs = variant.get("required_imports", [])
    for r in reqs:
      stmt = ""
      if isinstance(r, str):
        # Heuristic: if it doesn't look like a statement, treat as module name
        clean = r.strip()
        if clean.startswith("import") or clean.startswith("from"):
          stmt = clean
        else:
          stmt = f"import {clean}"
      elif isinstance(r, dict):
        mod = r.get("module")
        alias = r.get("alias")
        if mod:
          if alias:
            stmt = f"import {mod} as {alias}"
          else:
            stmt = f"import {mod}"

      # Use injection logic via context callback logic
      # (Accessing _callback_inject_preamble directly or via ctx)
      if stmt:
        self.ctx.inject_preamble(stmt)

  def _enter_scope(self) -> None:
    """Push a new scope onto the stack (e.g. entering a class or function)."""
    self._scope_stack.append(set())

  def _exit_scope(self) -> None:
    """Pop the current scope from the stack."""
    if len(self._scope_stack) > 1:
      self._scope_stack.pop()

  def _mark_stateful(self, var_name: str) -> None:
    """
    Marks a variable name as stateful in the current scope.

    Used for tracking Neural Layers to determine if calls should be rewritten
    as stateful invocations (e.g. `layer.apply(...)` instead of `layer(...)`).

    Args:
        var_name: The variable identifier (e.g., 'self.conv1').
    """
    self._scope_stack[-1].add(var_name)

  def _is_stateful(self, var_name: str) -> bool:
    """
    Checks if a variable is marked as stateful in any active scope.

    Traverses the scope stack from inner to outer.

    Args:
        var_name: The variable identifier.

    Returns:
        bool: True if the variable was previously marked as stateful.
    """
    for scope in reversed(self._scope_stack):
      if var_name in scope:
        return True
    return False

  def _report_failure(self, reason: str) -> None:
    """
    Records a fatal translation error for the current statement.
    This will trigger the Escape Hatch wrapper in `leave_SimpleStatementLine`.

    Args:
        reason: Human-readable error message.
    """
    self._current_stmt_errors.append(reason)

  def _report_warning(self, reason: str) -> None:
    """
    Records a non-fatal warning for the current statement.
    This wraps the statement in comments but preserves the transformed code.

    Args:
        reason: Human-readable warning message.
    """
    self._current_stmt_warnings.append(reason)

  def _get_qualified_name(self, node: cst.BaseExpression) -> Optional[str]:
    """
    Resolves a CST node to its fully qualified name using import aliases.

    Example:
        If ``import torch.nn as nn`` exists, ``nn.Linear`` resolves to ``torch.nn.Linear``.

    Args:
        node: The CST expression (Name or Attribute).

    Returns:
        Optional[str]: The resolved string (e.g. 'torch.abs') or None if unresolvable.
    """
    full_str = self._cst_to_string(node)
    if not full_str:
      return None

    parts = full_str.split(".")
    root = parts[0]

    if root in self._alias_map:
      canonical_root = self._alias_map[root]
      if len(parts) > 1:
        # e.g. root='nn' -> 'torch.nn', parts=['nn', 'Linear'] -> 'torch.nn.Linear'
        return f"{canonical_root}.{'.'.join(parts[1:])}"
      return canonical_root

    return full_str

  def _cst_to_string(self, node: cst.BaseExpression) -> Optional[str]:
    """
    Helper to flatten Attribute chains into strings.

    Args:
        node: The CST node to stringify.

    Returns:
        Optional[str]: Dotted string path (e.g. "a.b.c") or None if complex.
    """
    if isinstance(node, cst.Name):
      return node.value
    elif isinstance(node, cst.BinaryOperation):
      return type(node.operator).__name__
    elif isinstance(node, cst.Attribute):
      base = self._cst_to_string(node.value)
      if base:
        return f"{base}.{node.attr.value}"
    return None

  def _create_name_node(self, api_path: str) -> cst.BaseExpression:
    """
    Creates a LibCST node structure from a dotted string.

    Args:
        api_path: The fully qualified API name (e.g. 'jax.numpy.array').

    Returns:
        cst.BaseExpression: A nested Attribute (or Name) node used for AST replacement.
    """
    parts = api_path.split(".")
    node = cst.Name(parts[0])
    for part in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(part))
    return node

  def _create_dotted_name(self, name_str: str) -> Union[cst.Name, cst.Attribute]:
    """
    Alias for _create_name_node used by plugins.

    Args:
        name_str: Dotted string path.

    Returns:
        Union[cst.Name, cst.Attribute]: The generated node.
    """
    return self._create_name_node(name_str)

  def _get_mapping(self, name: str, silent: bool = False) -> Optional[Dict[str, Any]]:
    """
    Queries the SemanticsManager for the target framework's variant.
    Uses resolve_variant to handle framework inheritance (e.g. Pax -> JAX).

    Args:
        name: The fully qualified source name (e.g. 'torch.abs').
        silent: If True, suppresses failure reporting in strict mode.

    Returns:
        Optional[Dict[str, Any]]: The dictionary describing the target implementation,
                                  or None if not found/supported.
    """
    lookup = self.semantics.get_definition(name)
    if not lookup:
      if self.strict_mode and name.startswith(f"{self.source_fw}.") and not silent:
        self._report_failure(f"API '{name}' not found in semantics.")
      return None

    abstract_id, details = lookup

    # Check Verification Gating
    if not self.semantics.is_verified(abstract_id):
      if not silent:
        self._report_failure(f"Skipped '{name}': Marked unsafe by verification report.")
      return None

    # Retrieve Target Implementation via Inheritance aware resolver
    target_impl = self.semantics.resolve_variant(abstract_id, self.target_fw)

    if target_impl:
      get_tracer().log_match(source_api=name, target_api=target_impl.get("api", "Plugin Logic"), abstract_op=abstract_id)
    else:
      # Simple missing logic
      if self.strict_mode and not silent:
        self._report_failure(f"No mapping available for '{name}' -> '{self.target_fw}'")
      return None

    return target_impl

  def _is_docstring_node(self, node: cst.CSTNode, idx: int) -> bool:
    """
    Helper to detect module docstrings to ensure injection happens after them.

    Args:
        node: The statement node.
        idx: The index of the statement in the module body.

    Returns:
        bool: True if it is a docstring.
    """
    if idx != 0:
      return False
    if isinstance(node, cst.SimpleStatementLine) and len(node.body) == 1 and isinstance(node.body[0], cst.Expr):
      expr = node.body[0].value
      if isinstance(expr, (cst.SimpleString, cst.ConcatenatedString)):
        return True
    return False

  def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
    """
    Injects module-level preambles (e.g. Shim classes) requested by plugins.
    Ensures injection happens after docstrings to maintain valid Python help text.

    Args:
        original_node: Logic before transformation.
        updated_node: Logic after transformation.

    Returns:
        cst.Module: The module with injected preambles.
    """
    if not self._module_preamble:
      return updated_node

    # Parse injection strings into CST nodes
    new_stmts = []
    for stmt_code in self._module_preamble:
      try:
        parsed_mod = cst.parse_module(stmt_code)
        new_stmts.extend(parsed_mod.body)
      except cst.ParserSyntaxError:
        self._report_failure(f"Failed to inject module preamble: {stmt_code}")

    if not new_stmts:
      return updated_node

    # Determine insertion point (after docstring)
    body = list(updated_node.body)
    insert_idx = 0

    if body and self._is_docstring_node(body[0], 0):
      insert_idx = 1

    updated_body = body[:insert_idx] + new_stmts + body[insert_idx:]
    return updated_node.with_changes(body=updated_body)

  def visit_Import(self, node: cst.Import) -> Optional[bool]:
    """
    Scans ``import ...`` statements to populate the alias map.
    Example: ``import torch.nn as nn`` -> ``_alias_map['nn'] = 'torch.nn'``.

    Args:
        node: Import statement node.

    Returns:
        Optional[bool]: False to stop traversal of children.
    """
    for alias in node.names:
      full_name = self._cst_to_string(alias.name)
      if not full_name:
        continue

      if alias.asname:
        local_name = alias.asname.name.value
        self._alias_map[local_name] = full_name
      else:
        root = full_name.split(".")[0]
        self._alias_map[root] = root
    return False

  def visit_ImportFrom(self, node: cst.ImportFrom) -> Optional[bool]:
    """
    Scans ``from ... import ...`` statements to populate the alias map.
    Example: ``from torch import nn`` -> ``_alias_map['nn'] = 'torch.nn'``.

    Args:
        node: ImportFrom statement node.

    Returns:
        Optional[bool]: False to stop traversal of children.
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

      if alias.asname:
        local_name = alias.asname.name.value
      else:
        local_name = imported_name

      self._alias_map[local_name] = canonical_source

    return False

  def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> Optional[bool]:
    """
    Resets error tracking at the start of each line.
    Errors bubble up from children (Expressions) to this Statement handler.

    Args:
        node: The statement line node.

    Returns:
        Optional[bool]: True to continue traversal.
    """
    self._current_stmt_errors = []
    self._current_stmt_warnings = []
    return True

  def leave_SimpleStatementLine(
    self,
    original_node: cst.SimpleStatementLine,
    updated_node: cst.SimpleStatementLine,
  ) -> Union[cst.SimpleStatementLine, cst.FlattenSentinel]:
    """
    Handles error bubbling from expression rewrites.

    If errors occurred during processing of this line's children (e.g. failing
    to rewrite a function call), wrap the line in an ``EscapeHatch``.

    Prioritizes errors (reverting to original code) over warnings (using updated code).

    Args:
        original_node: The node before children were visited.
        updated_node: The node after children transformation.

    Returns:
        Union[cst.SimpleStatementLine, cst.FlattenSentinel]: The resulted node
        (possibly wrapped with comments).
    """
    if self._current_stmt_errors:
      unique_errors = list(dict.fromkeys(self._current_stmt_errors))
      message = "; ".join(unique_errors)
      # Revert to ORIGINAL node to ensure no partial mutations exist
      return EscapeHatch.mark_failure(original_node, message)

    if self._current_stmt_warnings:
      unique_warnings = list(dict.fromkeys(self._current_stmt_warnings))
      message = "; ".join(unique_warnings)
      # Warnings apply to UPDATED node
      return EscapeHatch.mark_failure(updated_node, message)

    return updated_node
