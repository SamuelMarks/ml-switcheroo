"""
Base Rewriter Implementation.

The foundation for the ``PivotRewriter``, aggregating mixins for:

- Resolution (Aliases)
- Scoping (State Tracking)
- Version Checking
- Error Handling

Also provides core infrastructure for:

- HookContext initialization.
- Global Preamble Injection.
- Knowledge Base Lookups (``_get_mapping``).
"""

from typing import Optional, List, Dict, Any, Union, Set
import libcst as cst

from ml_switcheroo.core.tracer import get_tracer
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.rewriter.types import SignatureContext

# Import Mixins
from ml_switcheroo.core.rewriter.resolver import ResolverMixin
from ml_switcheroo.core.rewriter.scopes import ScopingMixin
from ml_switcheroo.core.rewriter.ver_check import VersioningMixin
from ml_switcheroo.core.rewriter.errors import ErrorHandlingMixin


class BaseRewriter(
  ResolverMixin,
  ScopingMixin,
  ErrorHandlingMixin,
  VersioningMixin,
  cst.CSTTransformer,
):
  """
  The base class for AST transformation traversal.
  """

  def __init__(self, semantics: SemanticsManager, config: RuntimeConfig):
    """
    Initializes the rewriter and its mixins.
    """
    self.semantics = semantics
    self.config = config

    self.source_fw = str(config.effective_source)
    self.target_fw = str(config.effective_target)
    self.strict_mode = config.strict_mode

    # Initialize Hook Context
    self.ctx = HookContext(
      semantics,
      config,
      arg_injector=self._callback_inject_arg,
      preamble_injector=self._callback_inject_preamble,
    )

    # Initialize Mixin State
    self._current_stmt_errors: List[str] = []
    self._current_stmt_warnings: List[str] = []

    # ScopingMixin state
    self._scope_stack: List[Set[str]] = [set()]

    # ResolverMixin state
    self._alias_map: Dict[str, str] = {}

    # VersioningMixin state
    self._cached_target_version: Optional[str] = None
    self._version_checked = False

    # Additional State
    self._signature_stack: List[SignatureContext] = []
    self._in_module_class = False
    self._module_preamble: List[str] = []

    # --- FIX: Pre-populate Alias Map with Source Framework Aliases ---
    # This ensures strict mode and rewriters recognize implicit roots (like 'midl' for 'latex_dsl')
    # even if no explicit import statement is found in the AST.
    self._hydrate_source_aliases()

  def _hydrate_source_aliases(self) -> None:
    """
    Loads default aliases for the source framework from semantics config.
    """
    fw_conf = self.semantics.get_framework_config(self.source_fw)
    if fw_conf:
      alias_info = fw_conf.get("alias")
      # Handle Pydantic model or dict
      if hasattr(alias_info, "model_dump"):
        alias_info = alias_info.model_dump()

      if isinstance(alias_info, dict):
        name = alias_info.get("name")
        if name:
          # Map the alias (e.g. 'midl') to the full framework key ('latex_dsl')
          # or the module path?
          # The Resolver logic expects _alias_map values to be fully qualified prefixes.
          # Usually "torch" -> "torch". "midl" -> "latex_dsl"?
          # No, the semantics defines API paths.
          # if latex_dsl defines api="midl.Conv2d", then `midl` is the root.
          # We map alias -> alias to treat it as a known root.
          self._alias_map[name] = name

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
      # We check both the explicit framework name (e.g. 'torch.')
      # AND any known aliases in the map (e.g. 't.' or 'midl.').

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
    """
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
