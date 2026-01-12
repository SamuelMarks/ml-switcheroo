"""
API Logic Pass.

This module consolidates all API-level transformations, including:
1.  **Function Calls**: Remapping APIs (e.g., `torch.abs` -> `jnp.abs`), applying argument normalization,
    handling layout permutations, and executing transformation strategies (infix, lambda, macros).
2.  **Attributes**: Remapping attributes/constants (e.g., `torch.float32` -> `jnp.float32`).
3.  **Assignments**: Unwrapping functional return patterns (e.g., `layer.apply`).
4.  **Symbol Resolution**: Resolving aliases to fully qualified names for lookup.
5.  **Scoping**: Tracking stateful variables (layers) to inform call rewriting logic.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import re
import libcst as cst

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.core.hooks import get_hook
from ml_switcheroo.core.rewriter.calls.guards import apply_strict_guards
from ml_switcheroo.core.rewriter.calls.post import handle_post_processing
from ml_switcheroo.core.rewriter.calls.pre import handle_pre_checks, resolve_implicit_method
from ml_switcheroo.core.rewriter.calls.strategy import execute_strategy
from ml_switcheroo.core.rewriter.calls.utils import is_builtin, is_functional_apply, is_super_call, log_diff
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.interface import RewriterPass
from ml_switcheroo.core.rewriter.normalization_utils import convert_value_to_cst, extract_primitive_key
from ml_switcheroo.core.rewriter.types import SignatureContext
from ml_switcheroo.core.tracer import get_tracer
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.utils.node_diff import capture_node_source


class ApiPass(RewriterPass):
  """
  Transformation pass for rewiring API usage.

  Handles resolving function calls to Abstract Operations (The Hub) and then projecting
  them to the Target Framework (The Spoke). Also handles attribute renaming and
  stateful assignment tracking.
  """

  def transform(self, module: cst.Module, context: RewriterContext) -> cst.Module:
    """
    Executes the API transformation logic.

    Args:
        module: The source CST.
        context: Shared rewriter state.

    Returns:
        The transformed CST.
    """
    transformer = ApiTransformer(context)
    return module.visit(transformer)


class ApiTransformer(cst.CSTTransformer):
  """
  LibCST Transformer for API Logic.

  This class centralizes the logic for:
  - Resolving names/aliases.
  - Tracking scope/state.
  - Rewriting Calls, Attributes, and Assignments.
  """

  def __init__(self, context: RewriterContext) -> None:
    """
    Initialize the transformer.

    Args:
        context: The shared rewriter context.
    """
    self.context = context
    self._cached_source_traits: Optional[StructuralTraits] = None
    self._cached_target_traits: Optional[StructuralTraits] = None
    self._known_module_bases: Optional[Set[str]] = None

  # --- Properties & Helpers ---

  @property
  def semantics(self) -> SemanticsManager:
    """Accessor for semantics manager."""
    return self.context.semantics

  @property
  def config(self) -> RuntimeConfig:
    """Accessor for runtime config."""
    return self.context.config

  @property
  def source_fw(self) -> str:
    """Accessor for source framework key."""
    return self.context.source_fw

  @property
  def target_fw(self) -> str:
    """Accessor for target framework key."""
    return self.context.target_fw

  @property
  def strict_mode(self) -> bool:
    """Accessor for strict mode flag."""
    return self.config.strict_mode

  @property
  def source_traits(self) -> StructuralTraits:
    """Lazily loads source framework traits."""
    if self._cached_source_traits:
      return self._cached_source_traits

    config_dict = self.semantics.get_framework_config(self.source_fw)
    if config_dict and "traits" in config_dict:
      self._cached_source_traits = StructuralTraits.model_validate(config_dict["traits"])
    else:
      self._cached_source_traits = StructuralTraits()
    return self._cached_source_traits

  def _get_target_traits(self) -> StructuralTraits:
    """Lazily loads target framework traits."""
    if self._cached_target_traits:
      return self._cached_target_traits

    config_dict = self.semantics.get_framework_config(self.target_fw)
    if config_dict and "traits" in config_dict:
      self._cached_target_traits = StructuralTraits.model_validate(config_dict["traits"])
    else:
      self._cached_target_traits = StructuralTraits()

    return self._cached_target_traits

  def _get_source_lifecycle_lists(self) -> Tuple[Set[str], Set[str]]:
    """Returns strip and warn method sets for lifecycle management."""
    traits = self.source_traits
    return (
      set(traits.lifecycle_strip_methods),
      set(traits.lifecycle_warn_methods),
    )

  def _report_failure(self, reason: str) -> None:
    """Records a failure in the context error buffer."""
    self.context.current_stmt_errors.append(reason)

  def _report_warning(self, reason: str) -> None:
    """Records a warning in the context warning buffer."""
    self.context.current_stmt_warnings.append(reason)

  def _cst_to_string(self, node: cst.BaseExpression) -> Optional[str]:
    """Flattens CST nodes (Name/Attribute) to string."""
    if isinstance(node, cst.Name):
      return node.value
    elif isinstance(node, cst.Attribute):
      base = self._cst_to_string(node.value)
      if base:
        return f"{base}.{node.attr.value}"
    elif isinstance(node, cst.BinaryOperation):
      # Fallback for operators if visited (shouldn't happen often in this path)
      return type(node.operator).__name__
    return None

  def _get_qualified_name(self, node: cst.BaseExpression) -> Optional[str]:
    """Resolves aliases to get the Fully Qualified Name (FQN)."""
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

  def _create_name_node(self, api_path: str) -> cst.BaseExpression:
    """Constructs a CST node structure for a dotted API path."""
    parts = api_path.split(".")
    node = cst.Name(parts[0])
    for part in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(part))
    return node

  def _create_dotted_name(self, name_str: str) -> Union[cst.Name, cst.Attribute]:
    """Alias for create_name_node used by plugins."""
    # Type ignored because _create_name_node returns BaseExpression but plugins expect union subset
    return self._create_name_node(name_str)  # type: ignore

  def _get_mapping(self, name: str, silent: bool = False) -> Optional[Dict[str, Any]]:
    """Queries the Semantics Manager for the target implementation of the API."""
    lookup = self.semantics.get_definition(name)
    if not lookup:
      is_known_source_prefix = False
      root = name.split(".")[0]
      if root == self.source_fw or (self.context.alias_map and root in self.context.alias_map):
        is_known_source_prefix = True

      if self.strict_mode and is_known_source_prefix and not silent:
        self._report_failure(f"API '{name}' not found in semantics.")
      return None

    abstract_id, details = lookup

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

  def _handle_variant_imports(self, variant: Dict[str, Any]) -> None:
    """Injects required imports defined in the variant."""
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
        self.context.hook_context.inject_preamble(stmt)

  def check_version_constraints(self, min_v: Optional[str], max_v: Optional[str]) -> Optional[str]:
    """Checks if target version requirements are met."""
    if not min_v and not max_v:
      return None

    # Try Getting Version
    current = None
    fw_conf = self.semantics.get_framework_config(self.target_fw)
    if fw_conf and "version" in fw_conf:
      current = fw_conf["version"]
    else:
      import importlib.metadata

      pkg = self.target_fw
      if pkg == "flax_nnx":
        pkg = "flax"
      try:
        current = importlib.metadata.version(pkg)
      except Exception:
        pass

    if not current:
      return None

    def parse_v(v_str):
      parts = []
      # Fix: Use re module safely imported at global scope
      tokens = re.split(r"[^\d]+", v_str)
      for t in tokens:
        if t:
          parts.append(int(t))
      return tuple(parts)

    curr_tuple = parse_v(current)

    if min_v:
      if curr_tuple < parse_v(min_v):
        return f"Target {self.target_fw}@{current} is older than required {min_v}"

    if max_v:
      if curr_tuple >= parse_v(max_v):
        return f"Target {self.target_fw}@{current} exceeds max supported {max_v}"

    return None

  def _is_framework_base(self, name: str) -> bool:
    """Checks if a class name corresponds to any known framework Module base. Copied from Structural to support detection here."""
    if not name:
      return False

    if self._known_module_bases is None:
      self._known_module_bases = set()
      # Scan all registered configs for module_base traits
      for _, config in self.semantics.framework_configs.items():
        traits = config.get("traits")
        if traits:
          # Handle both dict and object access safely
          if isinstance(traits, dict):
            base = traits.get("module_base")
          else:
            base = getattr(traits, "module_base", None)

          if base:
            self._known_module_bases.add(base)

    if name in self._known_module_bases:
      return True

    # Suffix Check (e.g. 'nn.Module' matches 'torch.nn.Module')
    for known in self._known_module_bases:
      if known.endswith(f".{name}"):
        return True

    return False

  # --- Preamble Output ---
  # Use simple implementation for now without deduplication on identity which might be complex,
  # as simple set of strings works for identical injects.
  def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
    """
    Injects accumulated module-level preamble statements if they haven't been flushed yet
    by a prior pass (like StructuralPass). We deduplicate based on string content.
    """
    if not self.context.module_preamble:
      return updated_node

    new_stmts = []
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
    """Marks variable as stateful in current scope."""
    if self.context.scope_stack:
      self.context.scope_stack[-1].add(var_name)

  def _is_stateful(self, var_name: str) -> bool:
    """Checks if variable is stateful (traversing up scopes)."""
    for scope in reversed(self.context.scope_stack):
      if var_name in scope:
        return True
    return False

  # --- Context Stack Mirroring (Essential for Preamble Injection in Logic Plugins) ---

  def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
    """Enter class scope and detect Module."""
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
    """Exit class scope."""
    self.context.scope_stack.pop()
    if self.context.in_module_class:
      self.context.in_module_class = False
    return updated_node

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    """Enter function scope."""
    self.context.scope_stack.append(set())

    existing_args = set()
    for param in node.params.params:
      if isinstance(param.name, cst.Name):
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
    """
    Exit function scope.
    Flush any pending preamble statements requested by plugins during this pass.
    Also apply any pending signature injections (arguments).
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

  # --- Helper: Signature Injection (Restored from StructuralPass) ---
  def _inject_argument_to_signature(
    self, node: cst.FunctionDef, arg_name: str, annotation: Optional[str]
  ) -> cst.FunctionDef:
    """Injects a new argument after 'self' (or at start)."""
    params = list(node.params.params)
    insert_idx = 0
    if params and params[0].name.value == "self":
      insert_idx = 1

    # Avoid duplicate if already present
    if any(p.name.value == arg_name for p in params):
      return node

    anno_node = cst.Annotation(annotation=self._create_dotted_name(annotation)) if annotation else None

    # Ensure comma on previous arg
    if insert_idx > 0 and params[insert_idx - 1].comma == cst.MaybeSentinel.DEFAULT:
      params[insert_idx - 1] = params[insert_idx - 1].with_changes(
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
      )

    new_param = cst.Param(
      name=cst.Name(arg_name), annotation=anno_node, comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
    )
    params.insert(insert_idx, new_param)

    # Fix trailing comma structure
    if params:
      params[-1] = params[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

    new_params_node = node.params.with_changes(params=params)
    return node.with_changes(params=new_params_node)

  # --- Error Handling & Statement Processing ---

  def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> Optional[bool]:
    """Reset statement-level error buffers."""
    self.context.current_stmt_errors = []
    self.context.current_stmt_warnings = []
    return True

  def leave_SimpleStatementLine(
    self,
    original_node: cst.SimpleStatementLine,
    updated_node: cst.SimpleStatementLine,
  ) -> Union[cst.SimpleStatementLine, cst.FlattenSentinel]:
    """
    Check for errors generated by child expressions and wrap if needed.
    """
    if self.context.current_stmt_errors:
      unique_errors = list(dict.fromkeys(self.context.current_stmt_errors))
      message = "; ".join(unique_errors)
      return EscapeHatch.mark_failure(original_node, message)

    if self.context.current_stmt_warnings:
      unique_warnings = list(dict.fromkeys(self.context.current_stmt_warnings))
      message = "; ".join(unique_warnings)
      return EscapeHatch.mark_failure(updated_node, message)

    return updated_node

  # --- Resolver Logic ---

  def visit_Import(self, node: cst.Import) -> Optional[bool]:
    """Track import aliases."""
    for alias in node.names:
      full_name = self._cst_to_string(alias.name)
      if not full_name:
        continue

      if alias.asname:
        local_name = alias.asname.name.value
        self.context.alias_map[local_name] = full_name
      else:
        root = full_name.split(".")[0]
        self.context.alias_map[root] = root
    return False

  def visit_ImportFrom(self, node: cst.ImportFrom) -> Optional[bool]:
    """Track from-import aliases."""
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
      local_name = alias.asname.name.value if alias.asname else imported_name
      self.context.alias_map[local_name] = canonical_source

    return False

  # --- Attribute & Assignment Rewriting ---

  def leave_Assign(self, original_node: cst.Assign, updated_node: cst.Assign) -> cst.Assign:
    """
    Handle assignment rewriting.
    1. Track stateful initializations (e.g. self.layer = Linear...).
    2. Unwrap functional returns (e.g. y, state = layer.apply...).
    """
    # 1. Stateful Tracking
    if isinstance(original_node.value, cst.Call):
      func_name = self._get_qualified_name(original_node.value.func)
      if func_name:
        definition = self.semantics.get_definition(func_name)
        if definition:
          abstract_id, _ = definition
          origins = getattr(self.semantics, "_key_origins", {})
          tier = origins.get(abstract_id)
          if tier == SemanticTier.NEURAL.value:
            for target in original_node.targets:
              target_name = self._get_qualified_name(target.target)
              if target_name:
                if target_name.startswith("self.") and len(self.context.scope_stack) > 1:
                  # Track stateful variable in the class scope (parent of init scope)
                  self.context.scope_stack[-2].add(target_name)
                else:
                  self._mark_stateful(target_name)

    # 2. Assignment Unwrapping (Functional -> OOP)
    if isinstance(original_node.value, cst.Call):
      # Fix: Check property existence before access
      if hasattr(self, "source_traits"):
        traits = self.source_traits
      else:
        traits = StructuralTraits()

      unwrap_method = traits.functional_execution_method
      if is_functional_apply(original_node.value, unwrap_method):
        if len(updated_node.targets) == 1:
          target = updated_node.targets[0].target
          if isinstance(target, (cst.Tuple, cst.List)):
            elements = target.elements
            if len(elements) > 0:
              primary_target = elements[0].value
              new_target = cst.AssignTarget(target=primary_target)
              new_node = updated_node.with_changes(targets=[new_target])
              get_tracer().log_mutation(
                "Assignment Unwrapping", capture_node_source(original_node), capture_node_source(new_node)
              )
              return new_node

    return updated_node

  def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
    """
    Rewrites attributes and constants (e.g. torch.float32).
    Skips rewriting if the attribute looks like a function call (handled by leave_Call).
    """
    name = self._get_qualified_name(original_node)
    if not name:
      return updated_node

    lookup = self.semantics.get_definition(name)
    if lookup:
      _, details = lookup
      target_var = details.get("variants", {}).get(self.target_fw)

      # Plugin guard
      if target_var and "requires_plugin" in target_var:
        return updated_node

      # Function guard: If it has args, let leave_Call handle it
      if "std_args" in details and details["std_args"]:
        return updated_node

    # Perform mapping logic for constant/enum
    target_impl = self._get_mapping(name, silent=True)
    if target_impl and "api" in target_impl:
      self._handle_variant_imports(target_impl)
      return self._create_name_node(target_impl["api"])

    return updated_node

  # --- Call Rewriting ---

  def leave_Call(
    self, original_node: cst.Call, updated_node: cst.Call
  ) -> Union[cst.Call, cst.BinaryOperation, cst.UnaryOperation, cst.CSTNode]:
    """
    Main entry point for function call rewriting.
    """
    # 1. Identify Function
    func_name = self._get_qualified_name(original_node.func)

    # 2. Pre-Checks
    # Pass 'self' as rewriter interface (duck typing via properties)
    handled, result_node = handle_pre_checks(self, original_node, updated_node, func_name)
    if handled:
      return result_node

    # 3. Resolve Mapping
    mapping = self._get_mapping(func_name) if func_name else None

    # Fallback: Implicit Method
    if not mapping:
      guessed_name = resolve_implicit_method(self, original_node, func_name)
      if guessed_name:
        mapping = self._get_mapping(guessed_name, silent=True)
        if mapping:
          func_name = guessed_name

    if not mapping:
      if is_super_call(original_node):
        return updated_node

      if func_name and not is_builtin(func_name):
        get_tracer().log_inspection(node_str=func_name, outcome="Skipped", detail="No Entry in Semantics Knowledge Base")

      if self.strict_mode and func_name and func_name.startswith(f"{self.source_fw}."):
        self._report_failure(f"API '{func_name}' not found in semantics.")

      return updated_node

    # 4. Version Check
    min_v = mapping.get("min_version")
    max_v = mapping.get("max_version")
    v_warn = self.check_version_constraints(min_v, max_v)
    if v_warn:
      self._report_warning(v_warn)

    lookup = self.semantics.get_definition(func_name)
    if not lookup:
      return updated_node

    abstract_id, details = lookup

    if details.get("deprecated", False):
      msg = f"Usage of deprecated operation '{abstract_id}'."
      if details.get("replaced_by"):
        msg += f" Consider using '{details['replaced_by']}' instead."
      self._report_warning(msg)

    # 5. Execute Strategy
    result_node = execute_strategy(self, original_node, updated_node, mapping, details, abstract_id)

    # 6. Post Processing
    result_node = handle_post_processing(self, result_node, mapping, abstract_id)

    log_diff(f"Operation ({abstract_id})", original_node, result_node)
    return result_node

  # --- Argument Normalization (Helper required by Strategy) ---

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """Determines if a node is a module reference (not a variable)."""
    name = self._cst_to_string(node)
    if not name:
      return False

    if name in self.context.alias_map:
      return True

    known_roots = set()
    if self.config:
      known_roots.add(self.config.source_framework)
      known_roots.add(self.config.target_framework)
      if self.config.source_flavour:
        known_roots.add(self.config.source_flavour.split(".")[0])

    if self.semantics:
      configs = getattr(self.semantics, "framework_configs", {})
      for fw_key, conf in configs.items():
        known_roots.add(fw_key)
        alias_conf = conf.get("alias")
        if alias_conf and isinstance(alias_conf, dict):
          mod = alias_conf.get("module")
          if mod:
            known_roots.add(mod.split(".")[0])

    root = name.split(".")[0]
    return root in known_roots

  def _normalize_arguments(
    self,
    original_node: cst.Call,
    updated_node: cst.Call,
    op_details: Dict[str, Any],
    target_impl: Dict[str, Any],
  ) -> List[cst.Arg]:
    """
    Pivots arguments from source implementation -> Standard -> Target implementation.
    Handles renaming, reordering, and default injection.
    """
    # 1. Parse Standard Types
    std_args_raw = op_details.get("std_args", [])
    std_args_order = []
    defaults_map: Dict[str, Any] = {}
    variadic_arg_name = None

    for item in std_args_raw:
      if isinstance(item, dict):
        name = item.get("name")
        if name:
          std_args_order.append(name)
          if item.get("is_variadic"):
            variadic_arg_name = name
          if "default" in item:
            defaults_map[name] = item["default"]
      elif isinstance(item, (list, tuple)):
        std_args_order.append(item[0])
      else:
        std_args_order.append(item)

    # 2. Config Maps
    source_variant = op_details.get("variants", {}).get(self.source_fw, {}) or {}
    source_arg_map = source_variant.get("args", {}) or {}
    target_arg_map = target_impl.get("args", {}) or {}
    target_val_map = target_impl.get("arg_values", {}) or {}
    pack_target_kw = target_impl.get("pack_to_tuple")
    pack_as_type = target_impl.get("pack_as", "Tuple")
    target_inject_map = target_impl.get("inject_args", {}) or {}

    lib_to_std = {v: k for k, v in source_arg_map.items()}

    found_args: Dict[str, cst.Arg] = {}
    extra_args: List[cst.Arg] = []
    variadic_buffer: List[cst.Arg] = []

    # 3. Method Receiver Injection (e.g. x.add(y) -> add(x, y))
    is_method_call = isinstance(original_node.func, cst.Attribute)
    receiver_injected = False

    if is_method_call and self._is_module_alias(original_node.func.value):  # type: ignore
      is_method_call = False

    if is_method_call:
      if std_args_order:
        first_std_arg = std_args_order[0]
        arg_provided = False
        for arg in original_node.args:
          if arg.keyword:
            k_name = arg.keyword.value
            mapped = lib_to_std.get(k_name) or (k_name if k_name == first_std_arg else None)
            if mapped == first_std_arg:
              arg_provided = True
              break

        if not arg_provided:
          if isinstance(original_node.func, cst.Attribute):
            rec = original_node.func.value
            found_args[first_std_arg] = cst.Arg(value=rec)
            receiver_injected = True
      else:
        if isinstance(original_node.func, cst.Attribute):
          extra_args.append(cst.Arg(value=original_node.func.value))

    # 4. Process Args
    pos_idx = 1 if receiver_injected else 0
    packing_mode = False

    for i, upd_arg in enumerate(updated_node.args):
      if not upd_arg.keyword:
        if packing_mode:
          variadic_buffer.append(upd_arg)
        elif pos_idx < len(std_args_order):
          std_name = std_args_order[pos_idx]
          if pack_target_kw and std_name == variadic_arg_name:
            packing_mode = True
            variadic_buffer.append(upd_arg)
          else:
            if std_name not in found_args:
              found_args[std_name] = upd_arg
            pos_idx += 1
        else:
          extra_args.append(upd_arg)
      else:
        k_name = upd_arg.keyword.value
        std_name = lib_to_std.get(k_name, k_name)
        if std_name in std_args_order:
          found_args[std_name] = upd_arg
        else:
          extra_args.append(upd_arg)

    # 5. Pack Variadics
    if packing_mode and variadic_arg_name and pack_target_kw:
      elements = []
      for arg in variadic_buffer:
        elements.append(
          cst.Element(
            value=arg.value,
            comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
          )
        )

      is_list = pack_as_type == "List"
      if elements:
        trailing_comma = cst.MaybeSentinel.DEFAULT
        if not is_list and len(elements) == 1:
          trailing_comma = cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
        elements[-1] = elements[-1].with_changes(comma=trailing_comma)

      container_node = cst.List(elements=elements) if is_list else cst.Tuple(elements=elements)

      found_args[variadic_arg_name] = cst.Arg(
        keyword=cst.Name(pack_target_kw),
        value=container_node,
        equal=cst.AssignEqual(
          whitespace_before=cst.SimpleWhitespace(""),
          whitespace_after=cst.SimpleWhitespace(""),
        ),
      )

    # 6. Reconstruct
    new_args_list: List[cst.Arg] = []

    for std_name in std_args_order:
      if std_name not in found_args and std_name in defaults_map:
        try:
          default_val = defaults_map[std_name]
          lit_val_node = convert_value_to_cst(default_val)
          found_args[std_name] = cst.Arg(
            keyword=cst.Name(std_name),
            value=lit_val_node,
            equal=cst.AssignEqual(
              whitespace_before=cst.SimpleWhitespace(""),
              whitespace_after=cst.SimpleWhitespace(""),
            ),
          )
        except Exception:
          pass

      if std_name in found_args:
        current_arg = found_args[std_name]

        if std_name == variadic_arg_name and pack_target_kw:
          new_args_list.append(current_arg)
          continue

        tg_alias = target_arg_map.get(std_name, std_name)
        if tg_alias is None:
          continue

        final_val_node = current_arg.value

        if target_val_map and std_name in target_val_map:
          val_options = target_val_map[std_name]
          raw_key = extract_primitive_key(current_arg.value)
          if raw_key is not None and str(raw_key) in val_options:
            target_code = val_options[str(raw_key)]
            try:
              final_val_node = cst.parse_expression(target_code)
            except cst.ParserSyntaxError:
              pass

        should_use_keyword = current_arg.keyword is not None

        if should_use_keyword:
          new_arg = current_arg.with_changes(
            keyword=cst.Name(tg_alias),
            value=final_val_node,
            equal=cst.AssignEqual(
              whitespace_before=cst.SimpleWhitespace(""),
              whitespace_after=cst.SimpleWhitespace(""),
            ),
          )
          new_args_list.append(new_arg)
        else:
          if final_val_node is not current_arg.value:
            new_arg = current_arg.with_changes(value=final_val_node)
            new_args_list.append(new_arg)
          else:
            new_args_list.append(current_arg)

    # Append extras
    new_args_list.extend(extra_args)

    # 7. Inject Args
    if target_inject_map:
      for arg_name, arg_val in target_inject_map.items():
        if any(a.keyword and a.keyword.value == arg_name for a in new_args_list):
          continue

        val_node = convert_value_to_cst(arg_val)

        if len(new_args_list) > 0 and new_args_list[-1].comma == cst.MaybeSentinel.DEFAULT:
          new_args_list[-1] = new_args_list[-1].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

        injected_arg = cst.Arg(
          keyword=cst.Name(arg_name),
          value=val_node,
          equal=cst.AssignEqual(
            whitespace_before=cst.SimpleWhitespace(""),
            whitespace_after=cst.SimpleWhitespace(""),
          ),
        )
        new_args_list.append(injected_arg)

    # Formatting cleanups
    for i in range(len(new_args_list) - 1):
      new_args_list[i] = new_args_list[i].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    if new_args_list:
      new_args_list[-1] = new_args_list[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

    return new_args_list

  # --- Hook Accessors (Proxy) ---
  @property
  def ctx(self) -> Any:
    """Expose hook context for strategy invocation."""
    return self.context.hook_context

  # --- Preamble and Signature Logic ---
  # Note: ApiPass can inject arguments via context sig stack,
  # but structural application logic (injection into node) resides in leave_FunctionDef.

  def _apply_preamble(self, node: cst.FunctionDef, stmts_code: List[str]) -> cst.FunctionDef:
    """Injects source code statements at the start of the function body."""
    new_stmts = []
    for code in stmts_code:
      try:
        mod = cst.parse_module(code)
        new_stmts.extend(mod.body)
      except Exception:
        pass

    return self._inject_stmts_to_body(node, new_stmts)

  def _inject_stmts_to_body(self, node: cst.FunctionDef, new_stmts: List[cst.BaseStatement]) -> cst.FunctionDef:
    """Helper to insert statements respecting docstrings."""
    if isinstance(node.body, cst.SimpleStatementSuite):
      node = self._convert_to_indented_block(node)

    existing = list(node.body.body)
    idx = 0
    # Skip docstring if exists
    if existing and isinstance(existing[0], cst.SimpleStatementLine) and len(existing[0].body) == 1:
      expr = existing[0].body[0]
      if isinstance(expr, cst.Expr) and isinstance(expr.value, (cst.SimpleString, cst.ConcatenatedString)):
        idx = 1

    final_body = existing[:idx] + new_stmts + existing[idx:]
    return node.with_changes(body=node.body.with_changes(body=final_body))

  def _convert_to_indented_block(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """Unwraps simple one-liners to indented blocks for injection."""
    if isinstance(node.body, cst.SimpleStatementSuite):
      new_stmts = [cst.SimpleStatementLine(body=[s]) for s in node.body.body]
      return node.with_changes(body=cst.IndentedBlock(body=new_stmts))
    return node
