"""Helpers for ApiTransformer."""

from typing import Any

from typing import Optional, Union, List, Dict
import libcst as cst
import re
from ml_switcheroo.core.tracer import get_tracer


class ApiHelpersMixin:
  """Mixin providing CST string/node conversions and alias resolutions."""

  def _cst_to_string(self, node: cst.BaseExpression) -> Optional[str]:
    """Flattens CST nodes (Name/Attribute) to string."""
    if isinstance(node, cst.Name):
      return node.value
    elif isinstance(node, cst.Attribute):
      base = self._cst_to_string(node.value)
      if base:  # pragma: no cover
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

    if root in self.context.alias_map:  # type: ignore
      canonical_root = self.context.alias_map[root]  # type: ignore
      if len(parts) > 1:
        return f"{canonical_root}.{'.'.join(parts[1:])}"
      return str(canonical_root)  # pragma: no cover

    return full_str

  def _create_name_node(self, api_path: str) -> cst.BaseExpression:
    """Constructs a CST node structure for a dotted API path."""
    parts = api_path.split(".")
    node = cst.Name(parts[0])
    for part in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(part))  # type: ignore
    return node

  def _create_dotted_name(self, name_str: str) -> Union[cst.Name, cst.Attribute]:
    """Alias for create_name_node used by plugins."""
    # Type ignored because _create_name_node returns BaseExpression but plugins expect union subset
    return self._create_name_node(name_str)  # type: ignore

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """Determines if a node is a module reference (not a variable)."""
    name = self._cst_to_string(node)  # type: ignore
    if not name:
      return False

    if name in self.context.alias_map:  # type: ignore
      return True

    known_roots = set()
    if self.config:  # type: ignore  # pragma: no cover
      known_roots.add(self.config.source_framework)  # type: ignore
      known_roots.add(self.config.target_framework)  # type: ignore
      if self.config.source_flavour:  # type: ignore  # pragma: no cover
        known_roots.add(self.config.source_flavour.split(".")[0])  # type: ignore

    if self.semantics:  # type: ignore  # pragma: no cover
      configs = getattr(self.semantics, "framework_configs", {})  # type: ignore
      for fw_key, conf in configs.items():
        known_roots.add(fw_key)
        alias_conf = conf.get("alias")
        if alias_conf and isinstance(alias_conf, dict):
          mod = alias_conf.get("module")
          if mod:  # pragma: no cover
            known_roots.add(mod.split(".")[0])

    root = name.split(".")[0]
    return root in known_roots

  def _apply_preamble(self, node: cst.FunctionDef, stmts_code: List[str]) -> cst.FunctionDef:
    """Injects source code statements at the start of the function body."""
    new_stmts = []  # type: ignore
    for code in stmts_code:
      try:
        mod = cst.parse_module(code)
        new_stmts.extend(mod.body)
      except Exception:  # pragma: no cover
        pass  # pragma: no cover

    return self._inject_stmts_to_body(node, new_stmts)

  def _inject_stmts_to_body(self, node: cst.FunctionDef, new_stmts: List[cst.BaseStatement]) -> cst.FunctionDef:
    """Helper to insert statements respecting docstrings."""
    if isinstance(node.body, cst.SimpleStatementSuite):
      node = self._convert_to_indented_block(node)

    existing = list(node.body.body)
    idx = 0
    # Skip docstring if exists
    if existing and isinstance(existing[0], cst.SimpleStatementLine) and len(existing[0].body) == 1:  # pragma: no cover
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
    return node  # pragma: no cover

  def _get_mapping(self, name: str, silent: bool = False) -> Optional[Dict[str, Any]]:
    """Queries the Semantics Manager for the target implementation of the API."""
    lookup = self.semantics.get_definition(name)  # type: ignore
    if not lookup:
      is_known_source_prefix = False
      root = name.split(".")[0]
      if root == self.source_fw or (self.context.alias_map and root in self.context.alias_map):  # type: ignore  # pragma: no cover
        is_known_source_prefix = True

      if self.strict_mode and is_known_source_prefix and not silent:  # type: ignore  # pragma: no cover
        self._report_failure(f"API '{name}' not found in semantics.")  # type: ignore
      return None

    abstract_id, details = lookup

    if not self.semantics.is_verified(abstract_id):  # type: ignore
      if not silent:  # pragma: no cover
        self._report_failure(f"Skipped '{name}': Marked unsafe by verification report.")  # type: ignore
      return None

    target_impl = self.semantics.resolve_variant(abstract_id, self.target_fw)  # type: ignore

    if target_impl:
      get_tracer().log_match(
        source_api=name,
        target_api=target_impl.get("api", "Plugin Logic"),
        abstract_op=abstract_id,
      )
    else:
      if self.strict_mode and not silent:  # type: ignore  # pragma: no cover
        self._report_failure(f"No mapping available for '{name}' -> '{self.target_fw}'")  # type: ignore
      return None

    if isinstance(target_impl, dict):
      return target_impl
    return None  # pragma: no cover

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
      elif isinstance(r, dict):  # pragma: no cover
        mod = r.get("module")
        alias = r.get("alias")
        if mod:  # pragma: no cover
          if alias:
            stmt = f"import {mod} as {alias}"
          else:
            stmt = f"import {mod}"

      if stmt:  # pragma: no cover
        self.context.hook_context.inject_preamble(stmt)  # type: ignore

  def _is_framework_base(self, name: str) -> bool:
    """Checks if a class name corresponds to any known framework Module base."""
    if not name:
      return False

    if getattr(self, "_known_module_bases", None) is None:
      self._known_module_bases = set()
      for _, config in self.semantics.framework_configs.items():  # type: ignore
        traits = config.get("traits")
        if traits:
          base = traits.get("module_base") if isinstance(traits, dict) else getattr(traits, "module_base", None)
          if base:  # pragma: no cover
            self._known_module_bases.add(base)

    if name in self._known_module_bases:
      return True
    for known in self._known_module_bases:
      if known.endswith(f".{name}"):
        return True
    return False

  def check_version_constraints(self, min_v: Optional[str], max_v: Optional[str]) -> Optional[str]:
    """Checks if target version requirements are met."""
    if not min_v and not max_v:
      return None

    # Try Getting Version
    current = None
    fw_conf = self.semantics.get_framework_config(self.target_fw)  # type: ignore
    if fw_conf and "version" in fw_conf:
      current = fw_conf["version"]
    else:
      import importlib.metadata

      pkg = self.target_fw  # type: ignore
      if pkg == "flax_nnx":
        pkg = "flax"  # pragma: no cover
      try:
        current = importlib.metadata.version(pkg)
      except Exception:
        pass

    if not current:
      return None

    def parse_v(v_str: Any) -> Any:
      """Parses a version string into a tuple of integers."""
      parts = []
      # Fix: Use re module safely imported at global scope
      tokens = re.split(r"[^\d]+", v_str)
      for t in tokens:
        if t:  # pragma: no cover
          parts.append(int(t))
      return tuple(parts)

    curr_tuple = parse_v(current)

    if min_v:
      if curr_tuple < parse_v(min_v):
        return f"Target {self.target_fw}@{current} is older than required {min_v}"  # type: ignore

    if max_v:
      if curr_tuple >= parse_v(max_v):
        return f"Target {self.target_fw}@{current} exceeds max supported {max_v}"  # type: ignore

    return None

  def _inject_argument_to_signature(
    self,
    node: cst.FunctionDef,
    arg_name: str,
    annotation: Optional[str],
  ) -> cst.FunctionDef:
    """Injects a new argument after 'self' (or at start)."""
    params = list(node.params.params)
    insert_idx = 0
    if params and params[0].name.value == "self":  # pragma: no cover
      insert_idx = 1

    # Avoid duplicate if already present
    if any(p.name.value == arg_name for p in params):
      return node

    anno_node = cst.Annotation(annotation=self._create_dotted_name(annotation)) if annotation else None

    # Ensure comma on previous arg
    if insert_idx > 0 and params[insert_idx - 1].comma == cst.MaybeSentinel.DEFAULT:
      params[insert_idx - 1] = params[insert_idx - 1].with_changes(  # pragma: no cover
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))
      )

    new_param = cst.Param(
      name=cst.Name(arg_name),
      annotation=anno_node,
      comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
    )
    params.insert(insert_idx, new_param)

    # Fix trailing comma structure
    if params:  # pragma: no cover
      params[-1] = params[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

    new_params_node = node.params.with_changes(params=params)
    return node.with_changes(params=new_params_node)
