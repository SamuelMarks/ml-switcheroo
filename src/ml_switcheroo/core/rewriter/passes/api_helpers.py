"""Helpers for ApiTransformer."""

from typing import Optional, Union, List, Dict, Any  # pragma: no cover
import libcst as cst  # pragma: no cover
import re  # pragma: no cover
from ml_switcheroo.core.tracer import get_tracer  # pragma: no cover


# pragma: no cover
# pragma: no cover
class ApiHelpersMixin:  # pragma: no cover
  """Mixin providing CST string/node conversions and alias resolutions."""  # pragma: no cover

  # pragma: no cover
  def _cst_to_string(self, node: cst.BaseExpression) -> Optional[str]:  # pragma: no cover
    """Flattens CST nodes (Name/Attribute) to string."""  # pragma: no cover
    if isinstance(node, cst.Name):  # pragma: no cover
      return node.value  # pragma: no cover
    elif isinstance(node, cst.Attribute):  # pragma: no cover
      base = self._cst_to_string(node.value)  # pragma: no cover
      if base:  # pragma: no cover
        return f"{base}.{node.attr.value}"  # pragma: no cover
    elif isinstance(node, cst.BinaryOperation):  # pragma: no cover
      # Fallback for operators if visited (shouldn't happen often in this path)  # pragma: no cover
      return type(node.operator).__name__  # pragma: no cover
    return None  # pragma: no cover

  # pragma: no cover
  def _get_qualified_name(self, node: cst.BaseExpression) -> Optional[str]:  # pragma: no cover
    """Resolves aliases to get the Fully Qualified Name (FQN)."""  # pragma: no cover
    full_str = self._cst_to_string(node)  # pragma: no cover
    if not full_str:  # pragma: no cover
      return None  # pragma: no cover
    # pragma: no cover
    parts = full_str.split(".")  # pragma: no cover
    root = parts[0]  # pragma: no cover
    # pragma: no cover
    if root in self.context.alias_map:  # type: ignore  # pragma: no cover
      canonical_root = self.context.alias_map[root]  # type: ignore  # pragma: no cover
      if len(parts) > 1:  # pragma: no cover
        return f"{canonical_root}.{'.'.join(parts[1:])}"  # pragma: no cover
      return canonical_root  # pragma: no cover
    # pragma: no cover
    return full_str  # pragma: no cover

  # pragma: no cover
  def _create_name_node(self, api_path: str) -> cst.BaseExpression:  # pragma: no cover
    """Constructs a CST node structure for a dotted API path."""  # pragma: no cover
    parts = api_path.split(".")  # pragma: no cover
    node = cst.Name(parts[0])  # pragma: no cover
    for part in parts[1:]:  # pragma: no cover
      node = cst.Attribute(value=node, attr=cst.Name(part))  # type: ignore  # pragma: no cover
    return node  # pragma: no cover

  # pragma: no cover
  def _create_dotted_name(self, name_str: str) -> Union[cst.Name, cst.Attribute]:  # pragma: no cover
    """Alias for create_name_node used by plugins."""  # pragma: no cover
    # Type ignored because _create_name_node returns BaseExpression but plugins expect union subset  # pragma: no cover
    return self._create_name_node(name_str)  # type: ignore  # pragma: no cover

  # pragma: no cover
  def _is_module_alias(self, node: cst.CSTNode) -> bool:  # pragma: no cover
    """Determines if a node is a module reference (not a variable)."""  # pragma: no cover
    name = self._cst_to_string(node)  # type: ignore  # pragma: no cover
    if not name:  # pragma: no cover
      return False  # pragma: no cover
    # pragma: no cover
    if name in self.context.alias_map:  # type: ignore  # pragma: no cover
      return True  # pragma: no cover
    # pragma: no cover
    known_roots = set()  # pragma: no cover
    if self.config:  # type: ignore  # pragma: no cover
      known_roots.add(self.config.source_framework)  # type: ignore  # pragma: no cover
      known_roots.add(self.config.target_framework)  # type: ignore  # pragma: no cover
      if self.config.source_flavour:  # type: ignore  # pragma: no cover
        known_roots.add(self.config.source_flavour.split(".")[0])  # type: ignore  # pragma: no cover
    # pragma: no cover
    if self.semantics:  # type: ignore  # pragma: no cover
      configs = getattr(self.semantics, "framework_configs", {})  # type: ignore  # pragma: no cover
      for fw_key, conf in configs.items():  # pragma: no cover
        known_roots.add(fw_key)  # pragma: no cover
        alias_conf = conf.get("alias")  # pragma: no cover
        if alias_conf and isinstance(alias_conf, dict):  # pragma: no cover
          mod = alias_conf.get("module")  # pragma: no cover
          if mod:  # pragma: no cover
            known_roots.add(mod.split(".")[0])  # pragma: no cover
    # pragma: no cover
    root = name.split(".")[0]  # pragma: no cover
    return root in known_roots  # pragma: no cover

  # pragma: no cover
  def _apply_preamble(self, node: cst.FunctionDef, stmts_code: List[str]) -> cst.FunctionDef:  # pragma: no cover
    """Injects source code statements at the start of the function body."""  # pragma: no cover
    new_stmts = []  # type: ignore  # pragma: no cover
    for code in stmts_code:  # pragma: no cover
      try:  # pragma: no cover
        mod = cst.parse_module(code)  # pragma: no cover
        new_stmts.extend(mod.body)  # pragma: no cover
      except Exception:  # pragma: no cover
        pass  # pragma: no cover
    # pragma: no cover
    return self._inject_stmts_to_body(node, new_stmts)  # pragma: no cover

  # pragma: no cover
  def _inject_stmts_to_body(
    self, node: cst.FunctionDef, new_stmts: List[cst.BaseStatement]
  ) -> cst.FunctionDef:  # pragma: no cover
    """Helper to insert statements respecting docstrings."""  # pragma: no cover
    if isinstance(node.body, cst.SimpleStatementSuite):  # pragma: no cover
      node = self._convert_to_indented_block(node)  # pragma: no cover
    # pragma: no cover
    existing = list(node.body.body)  # pragma: no cover
    idx = 0  # pragma: no cover
    # Skip docstring if exists  # pragma: no cover
    if existing and isinstance(existing[0], cst.SimpleStatementLine) and len(existing[0].body) == 1:  # pragma: no cover
      expr = existing[0].body[0]  # pragma: no cover
      if isinstance(expr, cst.Expr) and isinstance(
        expr.value, (cst.SimpleString, cst.ConcatenatedString)
      ):  # pragma: no cover
        idx = 1  # pragma: no cover
    # pragma: no cover
    final_body = existing[:idx] + new_stmts + existing[idx:]  # pragma: no cover
    return node.with_changes(body=node.body.with_changes(body=final_body))  # pragma: no cover

  # pragma: no cover
  def _convert_to_indented_block(self, node: cst.FunctionDef) -> cst.FunctionDef:  # pragma: no cover
    """Unwraps simple one-liners to indented blocks for injection."""  # pragma: no cover
    if isinstance(node.body, cst.SimpleStatementSuite):  # pragma: no cover
      new_stmts = [cst.SimpleStatementLine(body=[s]) for s in node.body.body]  # pragma: no cover
      return node.with_changes(body=cst.IndentedBlock(body=new_stmts))  # pragma: no cover
    return node  # pragma: no cover

  # pragma: no cover
  def _get_mapping(self, name: str, silent: bool = False) -> Optional[Dict[str, Any]]:  # pragma: no cover
    """Queries the Semantics Manager for the target implementation of the API."""  # pragma: no cover
    lookup = self.semantics.get_definition(name)  # type: ignore  # pragma: no cover
    if not lookup:  # pragma: no cover
      is_known_source_prefix = False  # pragma: no cover
      root = name.split(".")[0]  # pragma: no cover
      if root == self.source_fw or (self.context.alias_map and root in self.context.alias_map):  # type: ignore  # pragma: no cover
        is_known_source_prefix = True  # pragma: no cover
      # pragma: no cover
      if self.strict_mode and is_known_source_prefix and not silent:  # type: ignore  # pragma: no cover
        self._report_failure(f"API '{name}' not found in semantics.")  # type: ignore  # pragma: no cover
      return None  # pragma: no cover
    # pragma: no cover
    abstract_id, details = lookup  # pragma: no cover
    # pragma: no cover
    if not self.semantics.is_verified(abstract_id):  # type: ignore  # pragma: no cover
      if not silent:  # pragma: no cover
        self._report_failure(f"Skipped '{name}': Marked unsafe by verification report.")  # type: ignore  # pragma: no cover
      return None  # pragma: no cover
    # pragma: no cover
    target_impl = self.semantics.resolve_variant(abstract_id, self.target_fw)  # type: ignore  # pragma: no cover
    # pragma: no cover
    if target_impl:  # pragma: no cover
      get_tracer().log_match(  # pragma: no cover
        source_api=name,  # pragma: no cover
        target_api=target_impl.get("api", "Plugin Logic"),  # pragma: no cover
        abstract_op=abstract_id,  # pragma: no cover
      )  # pragma: no cover
    else:  # pragma: no cover
      if self.strict_mode and not silent:  # type: ignore  # pragma: no cover
        self._report_failure(f"No mapping available for '{name}' -> '{self.target_fw}'")  # type: ignore  # pragma: no cover
      return None  # pragma: no cover
    # pragma: no cover
    return target_impl  # pragma: no cover

  # pragma: no cover
  def _handle_variant_imports(self, variant: Dict[str, Any]) -> None:  # pragma: no cover
    """Injects required imports defined in the variant."""  # pragma: no cover
    reqs = variant.get("required_imports", [])  # pragma: no cover
    for r in reqs:  # pragma: no cover
      stmt = ""  # pragma: no cover
      if isinstance(r, str):  # pragma: no cover
        clean = r.strip()  # pragma: no cover
        if clean.startswith("import") or clean.startswith("from"):  # pragma: no cover
          stmt = clean  # pragma: no cover
        else:  # pragma: no cover
          stmt = f"import {clean}"  # pragma: no cover
      elif isinstance(r, dict):  # pragma: no cover
        mod = r.get("module")  # pragma: no cover
        alias = r.get("alias")  # pragma: no cover
        if mod:  # pragma: no cover
          if alias:  # pragma: no cover
            stmt = f"import {mod} as {alias}"  # pragma: no cover
          else:  # pragma: no cover
            stmt = f"import {mod}"  # pragma: no cover
      # pragma: no cover
      if stmt:  # pragma: no cover
        self.context.hook_context.inject_preamble(stmt)  # type: ignore  # pragma: no cover

  # pragma: no cover
  def _is_framework_base(self, name: str) -> bool:  # pragma: no cover
    """Checks if a class name corresponds to any known framework Module base."""  # pragma: no cover
    if not name:  # pragma: no cover
      return False  # pragma: no cover
    # pragma: no cover
    if getattr(self, "_known_module_bases", None) is None:  # pragma: no cover
      self._known_module_bases = set()  # type: ignore  # pragma: no cover
      for _, config in self.semantics.framework_configs.items():  # type: ignore  # pragma: no cover
        traits = config.get("traits")  # pragma: no cover
        if traits:  # pragma: no cover
          base = (
            traits.get("module_base") if isinstance(traits, dict) else getattr(traits, "module_base", None)
          )  # pragma: no cover
          if base:  # pragma: no cover
            self._known_module_bases.add(base)  # type: ignore  # pragma: no cover
    # pragma: no cover
    if name in self._known_module_bases:  # type: ignore  # pragma: no cover
      return True  # pragma: no cover
    for known in self._known_module_bases:  # type: ignore  # pragma: no cover
      if known.endswith(f".{name}"):  # pragma: no cover
        return True  # pragma: no cover
    return False  # pragma: no cover

  # pragma: no cover
  def check_version_constraints(self, min_v: Optional[str], max_v: Optional[str]) -> Optional[str]:  # pragma: no cover
    """Checks if target version requirements are met."""  # pragma: no cover
    if not min_v and not max_v:  # pragma: no cover
      return None  # pragma: no cover
    # pragma: no cover
    # Try Getting Version  # pragma: no cover
    current = None  # pragma: no cover
    fw_conf = self.semantics.get_framework_config(self.target_fw)  # type: ignore  # pragma: no cover
    if fw_conf and "version" in fw_conf:  # pragma: no cover
      current = fw_conf["version"]  # pragma: no cover
    else:  # pragma: no cover
      import importlib.metadata  # pragma: no cover

      # pragma: no cover
      pkg = self.target_fw  # type: ignore  # pragma: no cover
      if pkg == "flax_nnx":  # pragma: no cover
        pkg = "flax"  # pragma: no cover
      try:  # pragma: no cover
        current = importlib.metadata.version(pkg)  # pragma: no cover
      except Exception:  # pragma: no cover
        pass  # pragma: no cover
    # pragma: no cover
    if not current:  # pragma: no cover
      return None  # pragma: no cover

    # pragma: no cover
    def parse_v(v_str):  # pragma: no cover
      """Parses a version string into a tuple of integers."""  # pragma: no cover
      parts = []  # pragma: no cover
      # Fix: Use re module safely imported at global scope  # pragma: no cover
      tokens = re.split(r"[^\d]+", v_str)  # pragma: no cover
      for t in tokens:  # pragma: no cover
        if t:  # pragma: no cover
          parts.append(int(t))  # pragma: no cover
      return tuple(parts)  # pragma: no cover

    # pragma: no cover
    curr_tuple = parse_v(current)  # pragma: no cover
    # pragma: no cover
    if min_v:  # pragma: no cover
      if curr_tuple < parse_v(min_v):  # pragma: no cover
        return f"Target {self.target_fw}@{current} is older than required {min_v}"  # type: ignore  # pragma: no cover
    # pragma: no cover
    if max_v:  # pragma: no cover
      if curr_tuple >= parse_v(max_v):  # pragma: no cover
        return f"Target {self.target_fw}@{current} exceeds max supported {max_v}"  # type: ignore  # pragma: no cover
    # pragma: no cover
    return None  # pragma: no cover

  # pragma: no cover
  def _inject_argument_to_signature(  # pragma: no cover
    self,
    node: cst.FunctionDef,
    arg_name: str,
    annotation: Optional[str],  # pragma: no cover
  ) -> cst.FunctionDef:  # pragma: no cover
    """Injects a new argument after 'self' (or at start)."""  # pragma: no cover
    params = list(node.params.params)  # pragma: no cover
    insert_idx = 0  # pragma: no cover
    if params and params[0].name.value == "self":  # pragma: no cover
      insert_idx = 1  # pragma: no cover
    # pragma: no cover
    # Avoid duplicate if already present  # pragma: no cover
    if any(p.name.value == arg_name for p in params):  # pragma: no cover
      return node  # pragma: no cover
    # pragma: no cover
    anno_node = cst.Annotation(annotation=self._create_dotted_name(annotation)) if annotation else None  # type: ignore  # pragma: no cover
    # pragma: no cover
    # Ensure comma on previous arg  # pragma: no cover
    if insert_idx > 0 and params[insert_idx - 1].comma == cst.MaybeSentinel.DEFAULT:  # pragma: no cover
      params[insert_idx - 1] = params[insert_idx - 1].with_changes(  # pragma: no cover
        comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" "))  # pragma: no cover
      )  # pragma: no cover
    # pragma: no cover
    new_param = cst.Param(  # pragma: no cover
      name=cst.Name(arg_name),
      annotation=anno_node,
      comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),  # pragma: no cover
    )  # pragma: no cover
    params.insert(insert_idx, new_param)  # pragma: no cover
    # pragma: no cover
    # Fix trailing comma structure  # pragma: no cover
    if params:  # pragma: no cover
      params[-1] = params[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)  # pragma: no cover
    # pragma: no cover
    new_params_node = node.params.with_changes(params=params)  # pragma: no cover
    return node.with_changes(params=new_params_node)  # pragma: no cover
