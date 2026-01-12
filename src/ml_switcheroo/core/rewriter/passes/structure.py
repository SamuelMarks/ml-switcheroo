"""
Structural Rewriting Pass.

This module consolidates all structural transformation logic, including:
1.  **Class Inheritance Rewriting**: Swapping framework base classes (e.g., ``torch.nn.Module`` -> ``flax.nnx.Module``).
2.  **Function Signature Rewriting**: Injecting or stripping state/context arguments (e.g., ``rngs``, ``ctx``).
3.  **Method Renaming**: Mapping lifecycle methods (e.g., ``forward`` -> ``__call__``).
4.  **Body Injection**: Handling ``super().__init__`` calls and preamble injection.
5.  **Type Annotation Rewriting**: Mapping framework-specific types (e.g., ``torch.Tensor`` -> ``jax.Array``).
"""

import re
import libcst as cst
from typing import Optional, Set, List, Tuple, Dict, Any, Union

from ml_switcheroo.core.rewriter.interface import RewriterPass
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.rewriter.types import SignatureContext
from ml_switcheroo.semantics.schema import StructuralTraits
from ml_switcheroo.core.escape_hatch import EscapeHatch
from ml_switcheroo.enums import SemanticTier


class StructuralPass(RewriterPass):
  """
  Pass responsible for modifying the structural scaffolding of the code.

  It transforms class definitions, function signatures, and type hints
  to match the idioms of the target framework.
  """

  def transform(self, module: cst.Module, context: RewriterContext) -> cst.Module:
    """
    Executes the structural transformation.

    Args:
        module: The source CST.
        context: Shared rewriter state.

    Returns:
        The transformed CST.
    """
    transformer = StructuralTransformer(context)
    return module.visit(transformer)


class StructuralTransformer(cst.CSTTransformer):
  """
  LibCST Transformer encapsulating all structural logic.

  Maintains internal state for scope depth, annotation context, and
  target framework traits.
  """

  def __init__(self, context: RewriterContext) -> None:
    """
    Initialize the transformer.

    Args:
        context: The shared rewriter context.
    """
    self.context = context
    self._in_annotation = False
    self._cached_target_traits: Optional[StructuralTraits] = None
    self._known_module_bases: Optional[Set[str]] = None

  # --- Properties & Helpers ---

  @property
  def target_traits(self) -> StructuralTraits:
    """Lazily load structural traits for the target framework."""
    if self._cached_target_traits:
      return self._cached_target_traits

    config = self.context.semantics.get_framework_config(self.context.target_fw)
    if config and "traits" in config:
      self._cached_target_traits = StructuralTraits.model_validate(config["traits"])
    else:
      self._cached_target_traits = StructuralTraits()
    return self._cached_target_traits

  def _get_target_tiers(self) -> List[str]:
    """Retrieve supported tiers (e.g. ['neural', 'array']) for the target."""
    config = self.context.semantics.get_framework_config(self.context.target_fw)
    if config and "tiers" in config:
      return config["tiers"]
    return [SemanticTier.ARRAY_API.value, SemanticTier.NEURAL.value, SemanticTier.EXTRAS.value]

  def _get_qualified_name(self, node: cst.BaseExpression) -> Optional[str]:
    """Resolves a CST node to a dotted string using context aliases."""
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
    """Helper to flatten Attribute chains."""
    if isinstance(node, cst.Name):
      return node.value
    elif isinstance(node, cst.Attribute):
      base = self._cst_to_string(node.value)
      if base:
        return f"{base}.{node.attr.value}"
    return None

  def _create_dotted_name(self, name_str: str) -> cst.BaseExpression:
    """Constructs a CST node sequence from a dotted string."""
    parts = name_str.split(".")
    node = cst.Name(parts[0])
    for part in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(part))
    return node

  def _is_framework_base(self, name: str) -> bool:
    """Checks if a class name corresponds to any known framework Module base."""
    if not name:
      return False

    if self._known_module_bases is None:
      self._known_module_bases = set()
      # Scan all registered configs for module_base traits
      for _, config in self.context.semantics.framework_configs.items():
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

  def _get_source_inference_methods(self) -> Set[str]:
    """Gets inference methods for source framework (e.g. forward)."""
    defaults = {"forward", "__call__", "call"}
    config = self.context.semantics.get_framework_config(self.context.source_fw)
    if config and "traits" in config:
      traits = StructuralTraits.model_validate(config["traits"])
      if traits.known_inference_methods:
        return traits.known_inference_methods
    return defaults

  def _get_type_mapping(self, name: str) -> Optional[Dict[str, Any]]:
    """Looks up type definition in semantics."""
    lookup = self.context.semantics.get_definition(name)
    if not lookup:
      return None
    abstract_id, _ = lookup
    return self.context.semantics.resolve_variant(abstract_id, self.context.target_fw)

  # --- Visitor Logic: Module Preamble Injection ---

  def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
    """
    Injects accumulated module-level preamble statements (e.g. imports, shim classes)
    requested by plugins during the rewrite.
    Flushes and clears the buffer to prevent double injection in subsequent passes.
    """
    if not self.context.module_preamble:
      return updated_node

    new_stmts = []
    # Deduplication now handled at insertion time in Context, so order is preserved.
    for code in self.context.module_preamble:
      try:
        mod = cst.parse_module(code)
        new_stmts.extend(mod.body)
      except Exception:
        pass

    # Clear buffer to prevent re-injection
    self.context.module_preamble.clear()

    if not new_stmts:
      return updated_node

    # Prepends to body (naive injection).
    # ImportFixer will tidy up imports later in the pipeline.
    return updated_node.with_changes(body=new_stmts + list(updated_node.body))

  # --- Visitor Logic: Types ---

  def visit_Annotation(self, node: cst.Annotation) -> Optional[bool]:
    """Flag entry into a type annotation context."""
    self._in_annotation = True
    return True

  def leave_Annotation(self, original_node: cst.Annotation, updated_node: cst.Annotation) -> cst.Annotation:
    """Flag exit from a type annotation context."""
    self._in_annotation = False
    return updated_node

  def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.BaseExpression:
    """Rewrite type names if inside an annotation."""
    if self._in_annotation:
      full_name = self._get_qualified_name(original_node)
      if full_name:
        mapping = self._get_type_mapping(full_name)
        if mapping and "api" in mapping:
          return self._create_dotted_name(mapping["api"])
    return updated_node

  def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
    """
    Rewrite dotted type attributes (e.g. torch.Tensor) if inside an annotation.
    This fixes issue where complex types like torch.Tensor fail to rewrite because
    leave_Name only handles the leaves individually without context.
    """
    if self._in_annotation:
      full_name = self._get_qualified_name(original_node)
      if full_name:
        mapping = self._get_type_mapping(full_name)
        if mapping and "api" in mapping:
          return self._create_dotted_name(mapping["api"])

    # NOTE: When used in mixed inheritance (PivotRewriter), we must ensure
    # the next class in MRO gets called if no change is made, or if we want chain logic.
    # But LibCST Transformers don't support super() chaining well on return values.
    # PivotRewriter wraps this. For standalone pass, this is final.
    if hasattr(super(), "leave_Attribute"):
      # If using multiple inheritance shim (PivotRewriter)
      return super().leave_Attribute(original_node, updated_node)  # type: ignore

    return updated_node

  # --- Visitor Logic: Classes ---

  def visit_ClassDef(self, node: cst.ClassDef) -> Optional[bool]:
    """Detect Module inheritance to set processing state."""
    self.context.scope_stack.append(set())

    is_module = False
    for base in node.bases:
      name = self._get_qualified_name(base.value)
      if name and self._is_framework_base(name):
        is_module = True
        break

    # Fallback: check raw name if alias resolution didn't catch it
    # e.g. 'nn.Module' without 'nn' being aliased in map
    if not is_module:
      for base in node.bases:
        raw_name = self._cst_to_string(base.value)
        if raw_name and self._is_framework_base(raw_name):
          is_module = True
          break

    if is_module:
      self.context.in_module_class = True
      # Verify target support
      supported_tiers = self._get_target_tiers()
      if SemanticTier.NEURAL.value not in supported_tiers and "neural" not in supported_tiers:
        self.context.current_stmt_errors.append(
          f"Target framework '{self.context.target_fw}' does not support Neural Network classes."
        )

    return True

  def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> Union[cst.ClassDef, cst.CSTNode]:
    """Rewrite class inheritance if necessary."""
    self.context.scope_stack.pop()

    if self.context.in_module_class:
      self.context.in_module_class = False

      # Check for errors bubbled up
      if self.context.current_stmt_errors:
        msg = "; ".join(self.context.current_stmt_errors)
        self.context.current_stmt_errors.clear()
        return EscapeHatch.mark_failure(original_node, msg)

      target_base = self.target_traits.module_base
      if target_base:
        new_bases = []
        for base in updated_node.bases:
          # Logic to identify if this specific base needs swapping
          # (only swap framework module bases)
          name = self._get_qualified_name(base.value)
          if not name:
            name = self._cst_to_string(base.value)

          if name and self._is_framework_base(name):
            new_bases.append(cst.Arg(value=self._create_dotted_name(target_base)))
            # Only swap the first valid base found to prevent duplication
            # unless multi-inheritance logic is needed
          else:
            new_bases.append(base)
        updated_node = updated_node.with_changes(bases=new_bases)

    return updated_node

  # --- Visitor Logic: Functions ---

  def visit_FunctionDef(self, node: cst.FunctionDef) -> Optional[bool]:
    """Push function context onto signature stack."""
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
    """Apply signature, name, and body writes to functions."""
    self.context.scope_stack.pop()
    if not self.context.signature_stack:
      return updated_node

    sig_ctx = self.context.signature_stack.pop()
    traits = self.target_traits

    # 1. Method Renaming
    if sig_ctx.is_module_method:
      curr_name = updated_node.name.value
      target_fwd = traits.forward_method
      known_fwds = self._get_source_inference_methods()

      if target_fwd and curr_name in known_fwds and curr_name != target_fwd:
        updated_node = updated_node.with_changes(name=cst.Name(target_fwd))

      if sig_ctx.is_init and traits.init_method_name and traits.init_method_name != "__init__":
        updated_node = updated_node.with_changes(name=cst.Name(traits.init_method_name))

    # 2. Magic Arguments & Super Init Logic
    if sig_ctx.is_init and sig_ctx.is_module_method:
      # Inject Magic Args (e.g. rngs)
      for arg_name, arg_type in traits.inject_magic_args:
        if arg_name not in sig_ctx.existing_args:
          found_injected = any(n == arg_name for n, _ in sig_ctx.injected_args)
          if not found_injected:
            sig_ctx.injected_args.append((arg_name, arg_type))

      # Strip Magic Args
      args_to_strip = set(traits.strip_magic_args)
      if traits.auto_strip_magic_args:
        args_to_strip.update(self.context.semantics.known_magic_args)
        native = {a[0] for a in traits.inject_magic_args}
        args_to_strip -= native

      for arg_name in args_to_strip:
        updated_node = self._strip_argument_from_signature(updated_node, arg_name)

      # Super Init Logic
      if traits.requires_super_init:
        updated_node = self._ensure_super_init(updated_node)
      else:
        updated_node = self._strip_super_init(updated_node)

    # 3. Apply Injection Queue
    # Params
    for name, annotation in sig_ctx.injected_args:
      updated_node = self._inject_argument_to_signature(updated_node, name, annotation)

    # Preamble Statements
    if sig_ctx.preamble_stmts:
      updated_node = self._apply_preamble(updated_node, sig_ctx.preamble_stmts)

    # Docstrings
    if sig_ctx.injected_args:
      updated_node = self._update_docstring(updated_node, sig_ctx.injected_args)

    return updated_node

  # --- Helpers: AST Mutation ---

  def _strip_argument_from_signature(self, node: cst.FunctionDef, arg_name: str) -> cst.FunctionDef:
    """Removes an argument by name from the function definition."""
    params = list(node.params.params)
    new_params = [p for p in params if not (isinstance(p.name, cst.Name) and p.name.value == arg_name)]
    return self._fix_comma(node, new_params)

  def _inject_argument_to_signature(
    self, node: cst.FunctionDef, arg_name: str, annotation: Optional[str]
  ) -> cst.FunctionDef:
    """Injects a new argument after 'self'."""
    params = list(node.params.params)
    insert_idx = 0
    if params and params[0].name.value == "self":
      insert_idx = 1

    anno_node = cst.Annotation(annotation=self._create_dotted_name(annotation)) if annotation else None
    new_param = cst.Param(name=cst.Name(arg_name), annotation=anno_node, comma=cst.MaybeSentinel.DEFAULT)

    params.insert(insert_idx, new_param)
    return self._fix_comma(node, params)

  def _fix_comma(self, node: cst.FunctionDef, params: List[cst.Param]) -> cst.FunctionDef:
    """Ensures logic commas are correct for argument lists."""
    for i in range(len(params) - 1):
      if params[i].comma == cst.MaybeSentinel.DEFAULT:
        params[i] = params[i].with_changes(comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")))

    if len(params) > 0:
      last = params[-1]
      if last.comma != cst.MaybeSentinel.DEFAULT:
        params[-1] = last.with_changes(comma=cst.MaybeSentinel.DEFAULT)

    new_params_node = node.params.with_changes(params=params)
    return node.with_changes(params=new_params_node)

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

  def _ensure_super_init(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """Injects super().__init__() call."""
    if self._has_super_init(node):
      return node
    stmt = cst.SimpleStatementLine(
      body=[
        cst.Expr(value=cst.Call(func=cst.Attribute(value=cst.Call(func=cst.Name("super")), attr=cst.Name("__init__"))))
      ]
    )
    return self._inject_stmts_to_body(node, [stmt])

  def _strip_super_init(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """Removes super().__init__() call."""
    if isinstance(node.body, cst.SimpleStatementSuite):
      return node
    if not hasattr(node.body, "body"):
      return node

    new_body = [s for s in node.body.body if not self._is_super_init_call(s)]
    return node.with_changes(body=node.body.with_changes(body=new_body))

  def _has_super_init(self, node: cst.FunctionDef) -> bool:
    """Checks for presence of super().__init__()."""
    if hasattr(node.body, "body"):
      for stmt in node.body.body:
        if self._is_super_init_call(stmt):
          return True
    return False

  def _is_super_init_call(self, stmt: cst.CSTNode) -> bool:
    """Detects super init pattern in a statement node."""
    if isinstance(stmt, cst.SimpleStatementLine) and len(stmt.body) == 1:
      expr_or_assign = stmt.body[0]
      val = expr_or_assign.value if hasattr(expr_or_assign, "value") else None
      if isinstance(val, cst.Call) and isinstance(val.func, cst.Attribute) and val.func.attr.value == "__init__":
        inner = val.func.value
        if isinstance(inner, cst.Call) and isinstance(inner.func, cst.Name) and inner.func.value == "super":
          return True
    return False

  def _update_docstring(self, node: cst.FunctionDef, args: List[Tuple[str, Optional[str]]]) -> cst.FunctionDef:
    """Appends argument descriptions to the docstring."""
    if not hasattr(node.body, "body") or not node.body.body:
      return node
    stmt = node.body.body[0]
    if not isinstance(stmt, cst.SimpleStatementLine) or len(stmt.body) != 1:
      return node
    expr = stmt.body[0]
    if not isinstance(expr, cst.Expr) or not isinstance(expr.value, cst.SimpleString):
      return node

    val = expr.value.value
    # Simple string manipulation to append args
    if '"""' in val:
      content = val.replace('"""', "")
      injection = "\n" + "\n".join([f"    {n}: Injected." for n, _ in args])
      new_val = f'"""{content}{injection}\n    """'
      new_expr = expr.with_changes(value=cst.SimpleString(new_val))
      new_stmt = stmt.with_changes(body=[expr.with_changes(value=new_expr)])
      return node.with_changes(body=node.body.with_changes(body=[new_stmt] + list(node.body.body[1:])))

    return node
