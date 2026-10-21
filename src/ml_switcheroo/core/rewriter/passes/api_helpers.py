"""Support for ApiTransformer.

This module provides helper utilities and mixin classes used by ApiTransformer
to perform CST string/node conversions, resolve API paths to FQNs, map APIs
to target frameworks, inject imports, and handle version check constraints.
"""

import re
import typing

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.core.tracer import get_tracer

if typing.TYPE_CHECKING:
  from ml_switcheroo.config import RuntimeConfig
  from ml_switcheroo.core.rewriter.context import RewriterContext
  from ml_switcheroo.semantics.manager import SemanticsManager


class ApiHelpersMixin:
  """Mixin providing CST string/node conversions and alias resolutions.

  This class contains utility methods for traversing, querying, and transforming
  CST nodes representing API calls, variables, modules, and signatures. It serves
  as a support mixin for the `ApiTransformer`, exposing methods for fully qualified
  name resolution, alias checking, target framework mappings, and version checks.
  """

  # Added for type checking
  context: "RewriterContext"

  if typing.TYPE_CHECKING:

    @property
    def config(self) -> "RuntimeConfig":
      """Return the runtime configuration.

      Returns:
          RuntimeConfig instance.
      """
      ...

    @property
    def semantics(self) -> "SemanticsManager":
      """Return the semantics manager.

      Returns:
          SemanticsManager instance.
      """
      ...

    @property
    def source_framework(self) -> str:
      """Return the source framework identifier.

      Returns:
          Source framework identifier.
      """
      ...

    @property
    def target_framework(self) -> str:
      """Return the target framework identifier.

      Returns:
          Target framework identifier.
      """
      ...

    @property
    def strict_mode(self) -> bool:
      """Return whether strict mode is enabled.

      Returns:
          True if strict mode is enabled.
      """
      ...

    def _report_failure(self, msg: str) -> None:
      """Record a failure message in the current statement error buffer.

      Args:
          msg: Failure description string.
      """
      ...

  def _cst_to_string(self, node: cst.CSTNode) -> str | None:
    """Flatten CST nodes (Name/Attribute) to string.

    Args:
        node (cst.CSTNode): The CST expression node to convert.

    Returns:
        Optional[str]: The flattened dotted string representation of the expression (e.g., 'foo.bar'),
        or None if the node type is not supported.
    """
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

  def _get_qualified_name(self, node: cst.BaseExpression) -> str | None:
    """Resolve aliases to get the Fully Qualified Name (FQN).

    Args:
        node (cst.BaseExpression): The CST expression node representing the name or attribute.

    Returns:
        Optional[str]: The resolved Fully Qualified Name (FQN) as a string, or None if resolving fails.
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
      return str(canonical_root)

    return full_str

  def _create_name_node(self, api_path: str) -> cst.Name | cst.Attribute:
    """Construct a CST node structure for a dotted API path.

    Args:
        api_path (str): The dotted API path string (e.g., 'foo.bar.baz').

    Returns:
        Union[cst.Name, cst.Attribute]: The constructed CST expression node representing the path.
    """
    parts = api_path.split(".")
    node: cst.Name | cst.Attribute = cst.Name(parts[0])
    for part in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(part))
    return node

  def _create_dotted_name(self, name_str: str) -> cst.Name | cst.Attribute:
    """Alias for create_name_node used by plugins.

    Args:
        name_str (str): The dotted name string to convert to a CST node structure.

    Returns:
        Union[cst.Name, cst.Attribute]: A CST Name or Attribute representing the dotted name.
    """
    return self._create_name_node(name_str)

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """Determine if a node is a module reference (not a variable).

    Args:
        node (cst.CSTNode): The CST node to inspect.

    Returns:
        bool: True if the node is identified as a module alias, False otherwise.
    """
    name = self._cst_to_string(node)
    if not name:
      return False

    if name in self.context.alias_map:
      return True

    known_roots = set()
    if self.config is not None:
      known_roots.add(self.config.source_framework)
      known_roots.add(self.config.target_framework)
      if self.config.source_flavour:
        known_roots.add(self.config.source_flavour.split(".")[0])

    if self.semantics is not None:
      configs = getattr(self.semantics, "framework_configs", {})
      for fw_key, conf in configs.items():
        known_roots.add(fw_key)
        alias_conf = conf.get("alias")
        if alias_conf and isinstance(alias_conf, dict):
          mod = alias_conf.get("module")
          if mod:
            known_roots.add(mod.split(".")[0])
          alias_name = alias_conf.get("name")
          if alias_name:
            known_roots.add(alias_name.split(".")[0])

    root = name.split(".")[0]
    return root in known_roots

  def _apply_preamble(self, node: cst.FunctionDef, stmts_code: list[str]) -> cst.FunctionDef:
    """Inject source code statements at the start of the function body.

    Args:
        node (cst.FunctionDef): The target function definition node.
        stmts_code (List[str]): A list of source code strings representing statements to inject.

    Returns:
        cst.FunctionDef: A new function definition node with the preamble statements injected.
    """
    new_stmts: list[cst.SimpleStatementLine | cst.BaseCompoundStatement] = []
    for code in stmts_code:
      try:
        mod = cst.parse_module(code)
        new_stmts.extend(mod.body)
      except (cst.ParserSyntaxError, Exception):
        pass

    return self._inject_stmts_to_body(node, new_stmts)

  def _inject_stmts_to_body(
    self,
    node: cst.FunctionDef,
    new_stmts: typing.Sequence[cst.SimpleStatementLine | cst.BaseCompoundStatement],
  ) -> cst.FunctionDef:
    """Support to insert statements respecting docstrings.

    Args:
        node (cst.FunctionDef): The function definition node where statements are injected.
        new_stmts (Sequence[Union[cst.SimpleStatementLine, cst.BaseCompoundStatement]]): The list of CST statements to insert into the function body.

    Returns:
        cst.FunctionDef: The updated function definition node with the statements inserted.
    """
    if isinstance(node.body, cst.SimpleStatementSuite):
      node = self._convert_to_indented_block(node)

    existing = list(node.body.body)
    idx = 0
    # Skip docstring if exists
    if existing and isinstance(existing[0], cst.SimpleStatementLine) and len(existing[0].body) == 1:
      expr = existing[0].body[0]
      if isinstance(expr, cst.Expr) and isinstance(expr.value, (cst.SimpleString, cst.ConcatenatedString)):
        idx = 1

    final_body = existing[:idx] + list(new_stmts) + existing[idx:]
    return node.with_changes(body=node.body.with_changes(body=final_body))

  def _convert_to_indented_block(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """Unwrap simple one-liners to indented blocks for injection.

    Args:
        node (cst.FunctionDef): The function definition node to convert.

    Returns:
        cst.FunctionDef: The updated function definition node with an IndentedBlock body.
    """
    if isinstance(node.body, cst.SimpleStatementSuite):
      new_stmts = [cst.SimpleStatementLine(body=[s]) for s in node.body.body]
      return node.with_changes(body=cst.IndentedBlock(body=new_stmts))
    return node

  def _get_mapping(self, name: str, silent: bool = False) -> dict[str, JsonValue] | None:
    """Query the Semantics Manager for the target implementation of the API.

    Args:
        name (str): The fully qualified name of the API to map.
        silent (bool, optional): If True, suppresses error/failure reports during lookup. Defaults to False.

    Returns:
        Optional[dict]: The target implementation mapping dictionary if found and verified, otherwise None.
    """
    lookup = self.semantics.get_definition(name)
    if not lookup:
      is_known_source_prefix = False
      root = name.split(".")[0]
      if root == self.source_framework or (self.context.alias_map and root in self.context.alias_map):
        is_known_source_prefix = True

      if self.strict_mode and is_known_source_prefix and not silent:
        self._report_failure(f"API '{name}' not found in semantics.")
      return None

    abstract_id, details = lookup

    if not self.semantics.is_verified(abstract_id):
      if not silent:
        self._report_failure(f"Skipped '{name}': Marked unsafe by verification report.")
      return None

    target_impl = self.semantics.resolve_variant(abstract_id, self.target_framework)

    if target_impl:
      raw_api = target_impl.get("api", "Plugin Logic")
      get_tracer().log_match(
        source_api=name,
        target_api=str(raw_api) if raw_api is not None else "Plugin Logic",
        abstract_op=abstract_id,
      )
    else:
      if self.strict_mode and not silent:
        from ml_switcheroo.core.rewriter.calls.utils import is_neural_on_math_backend

        origins = getattr(self.semantics, "_key_origins", {})
        tier = origins.get(abstract_id)
        if is_neural_on_math_backend(tier, self.target_framework):
          self._report_failure(
            f"Cannot map neural network abstraction '{name}' directly to pure math backend '{self.target_framework}'. Use a framework like Flax or Keras."
          )
        else:
          self._report_failure(f"No mapping available for '{name}' -> '{self.target_framework}'")
      return None

    impl_val = getattr({"v": target_impl}, "__getitem__")("v")
    if isinstance(impl_val, dict):
      return dict(impl_val)
    return None

  def _handle_variant_imports(self, variant: typing.Mapping[str, JsonValue | typing.Hashable | typing.Sized]) -> None:
    """Inject required imports defined in the variant mapping into the module context.

    Args:
        variant (Mapping[str, JsonValue]): The target variant dictionary mapping containing required imports.
    """
    for r in getattr(variant, "get")("required_imports", []):
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
        if isinstance(mod, str) and mod:
          if isinstance(alias, str) and alias:
            stmt = f"import {mod} as {alias}"
          else:
            stmt = f"import {mod}"

      if stmt:
        self.context.hook_context.inject_preamble(stmt)

  def _is_framework_base(self, name: str) -> bool:
    """Check if a class name corresponds to any known neural network framework Module base.

    Args:
        name (str): The name of the class to check.

    Returns:
        bool: True if the name matches a known framework module base (e.g., `nn.Module`), False otherwise.
    """
    if not name:
      return False

    known_bases: set[str] | None = getattr(self, "_known_module_bases", None)
    if known_bases is None:
      known_bases = set()
      for _, config in self.semantics.framework_configs.items():
        traits = config.get("traits")
        if traits:
          base = traits.get("module_base") if isinstance(traits, dict) else getattr(traits, "module_base", None)
          if isinstance(base, str) and base:
            known_bases.add(base)
      self._known_module_bases: set[str] = known_bases

    if name in self._known_module_bases:
      return True
    for known in self._known_module_bases:
      if known.endswith(f".{name}"):
        return True
    return False

  def check_version_constraints(self, min_v: str | None, max_v: str | None) -> str | None:
    """Check if target framework version requirements are met based on the mapping constraints.

    Args:
        min_v (Optional[str]): The minimum required version string, if any.
        max_v (Optional[str]): The maximum supported version string (exclusive), if any.

    Returns:
        Optional[str]: An error message string if version constraints are violated, or None if constraints are met.
    """
    if not min_v and not max_v:
      return None

    # Try Getting Version
    current: str | None = None
    fw_conf = self.semantics.get_framework_config(self.target_framework)
    if fw_conf and "version" in fw_conf and isinstance(fw_conf["version"], str):
      current = fw_conf["version"]
    else:
      import importlib.metadata

      pkg = self.target_framework
      if pkg == "flax_nnx":
        pkg = "flax"
      try:
        current = importlib.metadata.version(pkg)
      except (importlib.metadata.PackageNotFoundError, Exception):
        current = None

    if not current:
      return None

    def parse_v(v_str: str) -> tuple[int, ...]:
      """Parse a version string into a tuple of integers for comparison.

      Args:
          v_str (str): The version string (e.g., '1.2.3') or version-like object to parse.

      Returns:
          tuple: A tuple of integers representing the version segments.
      """
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
        return f"Target {self.target_framework}@{current} is older than required {min_v}"

    if max_v:
      if curr_tuple >= parse_v(max_v):
        return f"Target {self.target_framework}@{current} exceeds max supported {max_v}"

    return None

  def _inject_argument_to_signature(
    self,
    node: cst.FunctionDef,
    arg_name: str,
    annotation: str | None,
  ) -> cst.FunctionDef:
    """Inject a new argument after 'self' (or at the start if 'self' is not present).

    Args:
        node (cst.FunctionDef): The function definition node to modify.
        arg_name (str): The name of the argument to inject.
        annotation (Optional[str]): The optional type annotation string for the argument.

    Returns:
        cst.FunctionDef: The modified function definition node with the new argument injected.
    """
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
      name=cst.Name(arg_name),
      annotation=anno_node,
      comma=cst.Comma(whitespace_after=cst.SimpleWhitespace(" ")),
    )
    params.insert(insert_idx, new_param)

    # Fix trailing comma structure
    if params:
      params[-1] = params[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

    new_params_node = node.params.with_changes(params=params)
    return node.with_changes(params=new_params_node)
