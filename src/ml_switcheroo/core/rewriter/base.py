"""
Base Rewriter Implementation and Stage Abstraction.

This module defines the `RewriterStage` base class for pipeline passes and
maintains the `BaseRewriter` (shim) for backward compatibility with the
legacy monolithic structure.

It defines `RewriterProxy` to map property accessors from Mixins to the
shared `RewriterContext`.
"""

from typing import Optional, List, Dict, Any, Union, Set
import libcst as cst

from ml_switcheroo.analysis.symbol_table import SymbolTable
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.types import SignatureContext

# Import Mixins (Legacy aggregation support)
from ml_switcheroo.core.rewriter.resolver import ResolverMixin
from ml_switcheroo.core.rewriter.scopes import ScopingMixin
from ml_switcheroo.core.rewriter.ver_check import VersioningMixin
from ml_switcheroo.core.rewriter.errors import ErrorHandlingMixin


class RewriterProxy:
  """
  Mixin providing read/write accessors to the shared Context.
  Ensures mixins can access state via `self._attr` without knowing about `self.context`.
  """

  @property
  def context(self) -> RewriterContext:
    """Abstract required property."""
    raise NotImplementedError

  # --- Property Proxies to Context ---

  @property
  def semantics(self) -> SemanticsManager:
    return self.context.semantics

  @property
  def config(self) -> RuntimeConfig:
    return self.context.config

  @property
  def symbol_table(self) -> Optional[SymbolTable]:
    return self.context.symbol_table

  @property
  def ctx(self):
    """Expose hook context for plugins."""
    return self.context.hook_context

  @property
  def source_fw(self) -> str:
    return self.context.source_fw

  @property
  def target_fw(self) -> str:
    return self.context.target_fw

  @property
  def strict_mode(self) -> bool:
    return self.context.config.strict_mode

  # --- Mutable State Proxies (Get/Set) ---

  @property
  def _alias_map(self) -> Dict[str, str]:
    return self.context.alias_map

  @property
  def _scope_stack(self) -> List[Set[str]]:
    return self.context.scope_stack

  @property
  def _signature_stack(self) -> List[SignatureContext]:
    return self.context.signature_stack

  @property
  def _current_stmt_errors(self) -> List[str]:
    return self.context.current_stmt_errors

  @_current_stmt_errors.setter
  def _current_stmt_errors(self, value: List[str]):
    self.context.current_stmt_errors = value

  @property
  def _current_stmt_warnings(self) -> List[str]:
    return self.context.current_stmt_warnings

  @_current_stmt_warnings.setter
  def _current_stmt_warnings(self, value: List[str]):
    self.context.current_stmt_warnings = value

  @property
  def _in_module_class(self) -> bool:
    return self.context.in_module_class

  @_in_module_class.setter
  def _in_module_class(self, val: bool):
    self.context.in_module_class = val

  @property
  def _module_preamble(self) -> List[str]:
    return self.context.module_preamble


class RewriterStage(RewriterProxy, cst.CSTTransformer):
  """
  Abstract base class for a discrete rewriting pass.

  Operates on a shared `RewriterContext`.
  """

  def __init__(self, context: RewriterContext):
    """
    Initialize the stage.

    Args:
        context: The shared state object.
    """
    self._context = context

  @property
  def context(self) -> RewriterContext:
    return self._context


class BaseRewriter(
  ResolverMixin,
  ScopingMixin,
  ErrorHandlingMixin,
  VersioningMixin,
  RewriterStage,
):
  """
  Legacy monolithic base class.

  Acts as a compatibility shim connecting the old Mixin-based architecture
  to the new Context-based state storage.
  """

  def __init__(
    self,
    semantics_or_ctx: Union[SemanticsManager, RewriterContext],
    config: Optional[RuntimeConfig] = None,
    symbol_table: Optional[SymbolTable] = None,
  ):
    """
    Initializes the rewriter.

    Supports both legacy signature (separate args) and new signature (context object).

    Args:
        semantics_or_ctx: Either a SemanticsManager (legacy) or RewriterContext (new).
        config: RuntimeConfig (required if legacy init used).
        symbol_table: SymbolTable (optional).
    """
    if isinstance(semantics_or_ctx, RewriterContext):
      ctx = semantics_or_ctx
    else:
      if config is None:
        raise ValueError("Config required for legacy BaseRewriter initialization")
      # Create fresh context wrapping the provided components
      ctx = RewriterContext(
        semantics=semantics_or_ctx,
        config=config,
        symbol_table=symbol_table,
        # Callback binding happens here for self-reference
        arg_injector=None,
        preamble_injector=None,
      )

    super().__init__(ctx)

    # Late binding of callbacks because 'self' methods are available now
    # We need to bridge the inner context hooks to this instance methods if it's the root rewriter
    if self.context.hook_context is not None:
      # If context was created externally, hooks might already be bound.
      # If created here, they are None.
      # We rely on PivorRewriter or manual setup to bind if needed.
      pass

    # VersioningMixin state (maintained locally for caching optimization)
    self._cached_target_version: Optional[str] = None
    self._version_checked = False

  def _callback_inject_arg(self, name: str, annotation: Optional[str] = None) -> None:
    """
    Callback for plugins to inject arguments into the current function signature.
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
    Callback for plugins to inject statements at module or function level.
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

  # --- Shared Helpers ---

  def _handle_variant_imports(self, variant: Dict[str, Any]) -> None:
    """
    Processes ``required_imports`` from a variant specification.
    """
    reqs = variant.get("required_imports", [])
    for r in reqs:
      stmt = ""
      if isinstance(r, str):
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

      if stmt:
        self.ctx.inject_preamble(stmt)

  def _create_name_node(self, api_path: str) -> cst.BaseExpression:
    """
    Creates a LibCST node structure from a dotted string.
    """
    parts = api_path.split(".")
    node = cst.Name(parts[0])
    for part in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(part))
    return node

  def _create_dotted_name(self, name_str: str) -> Union[cst.Name, cst.Attribute]:
    """Alias for _create_name_node used by plugins."""
    return self._create_name_node(name_str)

  def _get_mapping(self, name: str, silent: bool = False) -> Optional[Dict[str, Any]]:
    """
    Queries the SemanticsManager for the target framework's variant.
    """
    lookup = self.semantics.get_definition(name)
    if not lookup:
      # Strict Mode Logic:
      # If the API starts with a known source prefix, we flag it as an error.
      is_known_source_prefix = False
      root = name.split(".")[0]

      if root == self.source_fw:
        is_known_source_prefix = True
      elif root in self._alias_map:
        is_known_source_prefix = True

      if self.strict_mode and is_known_source_prefix and not silent:
        self._report_failure(f"API '{name}' not found in semantics.")
      return None

    abstract_id, details = lookup

    # Check Verification Gating
    if not self.semantics.is_verified(abstract_id):
      if not silent:
        self._report_failure(f"Skipped '{name}': Marked unsafe by verification report.")
      return None

    target_impl = self.semantics.resolve_variant(abstract_id, self.target_fw)

    if target_impl:
      from ml_switcheroo.core.tracer import get_tracer

      get_tracer().log_match(
        source_api=name,
        target_api=target_impl.get("api", "Plugin Logic"),
        abstract_op=abstract_id,
      )
    else:
      if self.strict_mode and not silent:
        self._report_failure(f"No mapping available for '{name}' -> '{self.target_fw}'")
      return None

    return target_impl

  def _is_docstring_node(self, node: cst.CSTNode, idx: int) -> bool:
    """Helper to detect module docstrings."""
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
    This handles the root level modification logic.
    """
    # Call Mixin leave methods manually if not handled by MRO order in subclasses
    # BaseRewriter is legacy, usually mixins handle their own leave_Module.

    if not self._module_preamble:
      return updated_node

    new_stmts = []
    for stmt_code in self._module_preamble:
      try:
        parsed_mod = cst.parse_module(stmt_code)
        new_stmts.extend(parsed_mod.body)
      except cst.ParserSyntaxError:
        self._report_failure(f"Failed to inject module preamble: {stmt_code}")

    if not new_stmts:
      return updated_node

    body = list(updated_node.body)
    insert_idx = 0

    if body and self._is_docstring_node(body[0], 0):
      insert_idx = 1

    updated_body = body[:insert_idx] + new_stmts + body[insert_idx:]
    return updated_node.with_changes(body=updated_body)
