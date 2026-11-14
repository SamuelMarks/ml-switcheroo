"""State and Mapping Resolution Collaborators for ``ApiTransformer``.

Provides ``ApiPassState`` (holding scoped context and cached structural traits) and
``MappingResolver`` (performing CST string/node conversion, FQN alias resolution,
semantic variant lookup, preamble/argument injection, and version constraint checks).
"""

from __future__ import annotations

import importlib.metadata
import re
import typing
from dataclasses import dataclass

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.config import RuntimeConfig
from ml_switcheroo.core.hooks import HookContext
from ml_switcheroo.core.rewriter.calls.utils import is_neural_on_math_backend
from ml_switcheroo.core.rewriter.context import RewriterContext
from ml_switcheroo.core.tracer import get_tracer
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.semantics.schema import StructuralTraits

if typing.TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager


@typing.runtime_checkable
class _ModuleBaseProtocol(typing.Protocol):
  """Protocol for traits objects exposing ``module_base``.

  Attributes:
      module_base: Framework base module class path string, if configured.
  """

  module_base: str | None


@dataclass
class ApiPassState:
  """Typed shared state for ``ApiTransformer`` and its composed rewrite handlers.

  Attributes:
      context: Shared ``RewriterContext`` for the current transpilation pass.
      cached_source_traits: Lazily populated ``StructuralTraits`` for the source framework.
      cached_target_traits: Lazily populated ``StructuralTraits`` for the target framework.
      known_module_bases: Cached set of module base class names across registered frameworks.
  """

  context: RewriterContext
  cached_source_traits: StructuralTraits | None = None
  cached_target_traits: StructuralTraits | None = None
  known_module_bases: set[str] | None = None

  @property
  def semantics(self) -> "SemanticsManager":
    """Return the active ``SemanticsManager`` bound to ``self.context``.

    Returns:
        Active ``SemanticsManager`` instance.
    """
    return self.context.semantics

  @property
  def config(self) -> RuntimeConfig:
    """Return the active ``RuntimeConfig`` bound to ``self.context``.

    Returns:
        Active ``RuntimeConfig`` instance.
    """
    return self.context.config

  @property
  def source_framework(self) -> str:
    """Return the effective source framework key.

    Returns:
        Source framework string identifier.
    """
    return str(self.context.source_framework)

  @property
  def target_framework(self) -> str:
    """Return the effective target framework key.

    Returns:
        Target framework string identifier.
    """
    return str(self.context.target_framework)

  @property
  def strict_mode(self) -> bool:
    """Return whether strict unmapped-API validation is enabled.

    Returns:
        ``True`` if strict mode is active, otherwise ``False``.
    """
    return self.config.strict_mode

  @property
  def ctx(self) -> HookContext:
    """Return the ``HookContext`` exposed to plugin strategies.

    Returns:
        Bound ``HookContext`` instance.
    """
    return self.context.hook_context

  @property
  def source_traits(self) -> StructuralTraits:
    """Lazily resolve and cache ``StructuralTraits`` for ``self.source_framework``.

    Returns:
        Resolved ``StructuralTraits`` for the source framework.
    """
    if self.cached_source_traits is not None:
      return self.cached_source_traits

    from ml_switcheroo.semantics.resolution import query_framework_traits

    fw_traits = query_framework_traits(self.semantics, self.source_framework)
    self.cached_source_traits = fw_traits.traits if fw_traits.traits is not None else StructuralTraits()
    return self.cached_source_traits

  @property
  def target_traits(self) -> StructuralTraits:
    """Lazily resolve and cache ``StructuralTraits`` for ``self.target_framework``.

    Returns:
        Resolved ``StructuralTraits`` for the target framework.
    """
    if self.cached_target_traits is not None:
      return self.cached_target_traits

    from ml_switcheroo.semantics.resolution import query_framework_traits

    fw_traits = query_framework_traits(self.semantics, self.target_framework)
    self.cached_target_traits = fw_traits.traits if fw_traits.traits is not None else StructuralTraits()
    return self.cached_target_traits

  def get_source_lifecycle_lists(self) -> tuple[set[str], set[str]]:
    """Return strip and warning lifecycle method sets for the source framework.

    Returns:
        Tuple of ``(strip_methods, warn_methods)``.
    """
    traits = self.source_traits
    return (set(traits.lifecycle_strip_methods), set(traits.lifecycle_warn_methods))

  def report_failure(self, reason: str) -> None:
    """Record a failure diagnostic on the current statement buffer.

    Args:
        reason: Failure description message.
    """
    self.context.current_stmt_errors.append(reason)

  def report_warning(self, reason: str) -> None:
    """Record a warning diagnostic on the current statement buffer.

    Args:
        reason: Warning description message.
    """
    self.context.current_stmt_warnings.append(reason)

  def mark_stateful(self, var_name: str) -> None:
    """Mark ``var_name`` as a stateful layer/object in the innermost scope.

    Args:
        var_name: Variable identifier to mark stateful.
    """
    if self.context.scope_stack:
      self.context.scope_stack[-1].add(var_name)

  def is_stateful(self, var_name: str) -> bool:
    """Return whether ``var_name`` is marked stateful in any enclosing lexical scope.

    Args:
        var_name: Variable identifier to check.

    Returns:
        ``True`` if ``var_name`` is stateful in an active scope.
    """
    return any(var_name in scope for scope in reversed(self.context.scope_stack))


class MappingResolver:
  """Collaborator resolving CST expressions, aliases, semantic mappings, and signature injections.

  Attributes:
      state: Shared ``ApiPassState`` instance.
  """

  def __init__(self, state: ApiPassState) -> None:
    """Initialize the mapping resolver over ``state``.

    Args:
        state: Shared ``ApiPassState`` instance.
    """
    self.state: ApiPassState = state

  def cst_to_string(self, node: cst.CSTNode) -> str | None:
    """Flatten ``Name`` / ``Attribute`` / ``BinaryOperation`` CST nodes into a dotted string.

    Args:
        node: CST expression node to flatten.

    Returns:
        Flattened dotted string representation or ``None`` if unsupported.
    """
    if isinstance(node, cst.Name):
      return node.value
    if isinstance(node, cst.Attribute):
      base = self.cst_to_string(node.value)
      if base:
        return f"{base}.{node.attr.value}"
    elif isinstance(node, cst.BinaryOperation):
      return type(node.operator).__name__
    return None

  def get_qualified_name(
    self,
    node: cst.BaseExpression,
    *,
    cst_to_string_fn: typing.Callable[[cst.CSTNode], str | None] | None = None,
  ) -> str | None:
    """Resolve import aliases to produce the fully qualified API name for ``node``.

    Args:
        node: CST expression node representing a name or attribute chain.
        cst_to_string_fn: Optional override callback used to flatten ``node``.

    Returns:
        Resolved fully qualified name string, or ``None`` if ``node`` cannot be flattened.
    """
    flattener = cst_to_string_fn if cst_to_string_fn is not None else self.cst_to_string
    full_str = flattener(node)
    if not full_str:
      return None

    parts = full_str.split(".")
    root = parts[0]

    if root in self.state.context.alias_map:
      canonical_root = self.state.context.alias_map[root]
      if len(parts) > 1:
        return f"{canonical_root}.{'.'.join(parts[1:])}"
      return str(canonical_root)

    return full_str

  def create_name_node(self, api_path: str) -> cst.Name | cst.Attribute:
    """Construct a CST ``Name`` or ``Attribute`` chain for ``api_path``.

    Args:
        api_path: Dotted API path string (e.g. ``'jnp.abs'``).

    Returns:
        Constructed ``cst.Name`` or ``cst.Attribute`` node.
    """
    parts = api_path.split(".")
    node: cst.Name | cst.Attribute = cst.Name(parts[0])
    for part in parts[1:]:
      node = cst.Attribute(value=node, attr=cst.Name(part))
    return node

  def create_dotted_name(self, name_str: str) -> cst.Name | cst.Attribute:
    """Construct a CST ``Name`` or ``Attribute`` chain for ``name_str``.

    Args:
        name_str: Dotted name string.

    Returns:
        Constructed ``cst.Name`` or ``cst.Attribute`` node.
    """
    return self.create_name_node(name_str)

  def is_module_alias(
    self,
    node: cst.CSTNode,
    *,
    cst_to_string_fn: typing.Callable[[cst.CSTNode], str | None] | None = None,
  ) -> bool:
    """Return whether ``node`` references an imported module alias or framework root.

    Args:
        node: CST node to inspect.
        cst_to_string_fn: Optional override callback used to flatten ``node``.

    Returns:
        ``True`` if ``node`` resolves to a module alias or known framework root.
    """
    flattener = cst_to_string_fn if cst_to_string_fn is not None else self.cst_to_string
    name = flattener(node)
    if not name:
      return False

    if name in self.state.context.alias_map:
      return True

    known_roots: set[str] = set()
    cfg = self.state.config
    if cfg is not None:
      known_roots.add(cfg.source_framework)
      known_roots.add(cfg.target_framework)
      if cfg.source_flavour:
        known_roots.add(cfg.source_flavour.split(".")[0])

    sem = self.state.semantics
    if sem is not None:
      try:
        configs = sem.framework_configs
      except AttributeError:
        configs = {}
      for fw_key, conf in configs.items():
        known_roots.add(fw_key)
        alias_conf = conf.get("alias")
        if alias_conf and isinstance(alias_conf, dict):
          mod = alias_conf.get("module")
          if isinstance(mod, str) and mod:
            known_roots.add(mod.split(".")[0])
          alias_name = alias_conf.get("name")
          if isinstance(alias_name, str) and alias_name:
            known_roots.add(alias_name.split(".")[0])

    root = name.split(".")[0]
    return root in known_roots

  @staticmethod
  def _parse_preamble_snippet(
    code: str,
  ) -> typing.Sequence[cst.SimpleStatementLine | cst.BaseCompoundStatement]:
    """Parse a single preamble source string into CST body statements.

    Args:
        code: Python statement source string to parse.

    Returns:
        Sequence of parsed CST statements, or an empty tuple if syntax is invalid.
    """
    try:
      return cst.parse_module(code).body
    except cst.ParserSyntaxError:
      return ()

  def apply_preamble(self, node: cst.FunctionDef, stmts_code: list[str]) -> cst.FunctionDef:
    """Parse and inject source statement strings at the beginning of ``node``'s body.

    Args:
        node: Target ``cst.FunctionDef`` node.
        stmts_code: Python statement strings to parse and inject.

    Returns:
        Updated ``cst.FunctionDef`` with preamble statements prepended after any docstring.
    """
    new_stmts: list[cst.SimpleStatementLine | cst.BaseCompoundStatement] = []
    for code in stmts_code:
      new_stmts.extend(self._parse_preamble_snippet(code))

    return self.inject_stmts_to_body(node, new_stmts)

  def inject_stmts_to_body(
    self,
    node: cst.FunctionDef,
    new_stmts: typing.Sequence[cst.SimpleStatementLine | cst.BaseCompoundStatement],
  ) -> cst.FunctionDef:
    """Insert ``new_stmts`` into ``node``'s body immediately after any leading docstring.

    Args:
        node: Target ``cst.FunctionDef`` node.
        new_stmts: Sequence of CST statements to insert.

    Returns:
        Updated ``cst.FunctionDef`` node.
    """
    if isinstance(node.body, cst.SimpleStatementSuite):
      node = self.convert_to_indented_block(node)

    existing = list(node.body.body)
    idx = 0
    if existing and isinstance(existing[0], cst.SimpleStatementLine) and len(existing[0].body) == 1:
      expr = existing[0].body[0]
      if isinstance(expr, cst.Expr) and isinstance(expr.value, (cst.SimpleString, cst.ConcatenatedString)):
        idx = 1

    final_body = existing[:idx] + list(new_stmts) + existing[idx:]
    return node.with_changes(body=node.body.with_changes(body=final_body))

  def convert_to_indented_block(self, node: cst.FunctionDef) -> cst.FunctionDef:
    """Convert a one-liner ``SimpleStatementSuite`` function body into an ``IndentedBlock``.

    Args:
        node: ``cst.FunctionDef`` node to normalize.

    Returns:
        ``cst.FunctionDef`` with an ``IndentedBlock`` body.
    """
    if isinstance(node.body, cst.SimpleStatementSuite):
      new_stmts = [cst.SimpleStatementLine(body=[s]) for s in node.body.body]
      return node.with_changes(body=cst.IndentedBlock(body=new_stmts))
    return node

  def get_mapping(
    self,
    name: str,
    silent: bool = False,
    *,
    report_failure_fn: typing.Callable[[str], None] | None = None,
  ) -> dict[str, JsonValue] | None:
    """Query ``SemanticsManager`` for the verified target variant mapping of ``name``.

    Args:
        name: Fully qualified source API name.
        silent: When ``True``, suppresses failure reporting on unmapped or unverified APIs.
        report_failure_fn: Optional callback override used to record failure messages.

    Returns:
        Target variant mapping dictionary if found and verified, else ``None``.
    """
    reporter = report_failure_fn if report_failure_fn is not None else self.state.report_failure
    sem = self.state.semantics
    lookup = sem.get_definition(name)
    if not lookup:
      is_known_source_prefix = False
      root = name.split(".")[0]
      if root == self.state.source_framework or (self.state.context.alias_map and root in self.state.context.alias_map):
        is_known_source_prefix = True

      if self.state.strict_mode and is_known_source_prefix and not silent:
        reporter(f"API '{name}' not found in semantics.")
      return None

    abstract_id, _ = lookup

    if not sem.is_verified(abstract_id):
      if not silent:
        reporter(f"Skipped '{name}': Marked unsafe by verification report.")
      return None

    target_impl = sem.resolve_variant(abstract_id, self.state.target_framework)

    if isinstance(target_impl, dict) and target_impl:
      raw_api = target_impl.get("api", "Plugin Logic")
      get_tracer().log_match(
        source_api=name,
        target_api=str(raw_api) if raw_api is not None else "Plugin Logic",
        abstract_op=abstract_id,
      )
      return dict(target_impl)

    if self.state.strict_mode and not silent:
      try:
        origins = sem._key_origins
      except AttributeError:
        origins = {}
      tier = origins.get(abstract_id)
      if is_neural_on_math_backend(tier, self.state.target_framework):
        reporter(
          f"Cannot map neural network abstraction '{name}' directly to pure math backend "
          f"'{self.state.target_framework}'. Use a framework like Flax or Keras."
        )
      else:
        reporter(f"No mapping available for '{name}' -> '{self.state.target_framework}'")
    return None

  def handle_variant_imports(self, variant: typing.Mapping[str, RuntimeTensorOrValue]) -> None:
    """Inject ``required_imports`` declared on ``variant`` into the active ``HookContext``.

    Args:
        variant: Target variant mapping dictionary.
    """
    raw_reqs = variant.get("required_imports", [])
    if not isinstance(raw_reqs, list):
      return
    for r in raw_reqs:
      stmt = ""
      if isinstance(r, str):
        clean = r.strip()
        stmt = clean if (clean.startswith(("import", "from"))) else f"import {clean}"
      elif isinstance(r, dict):
        mod = r.get("module")
        alias = r.get("alias")
        if isinstance(mod, str) and mod:
          stmt = f"import {mod} as {alias}" if (isinstance(alias, str) and alias) else f"import {mod}"

      if stmt:
        self.state.context.hook_context.inject_preamble(stmt)

  def is_framework_base(self, name: str) -> bool:
    """Return whether ``name`` matches any registered framework neural module base class.

    Args:
        name: Candidate base class name string.

    Returns:
        ``True`` if ``name`` matches a known framework module base.
    """
    if not name:
      return False

    known_bases = self.state.known_module_bases
    if known_bases is None:
      known_bases = set()
      for config in self.state.semantics.framework_configs.values():
        traits = config.get("traits")
        if traits:
          base = (
            traits.get("module_base")
            if isinstance(traits, dict)
            else (traits.module_base if isinstance(traits, _ModuleBaseProtocol) else None)
          )
          if isinstance(base, str) and base:
            known_bases.add(base)
      self.state.known_module_bases = known_bases

    if name in known_bases:
      return True
    return any(known.endswith(f".{name}") for known in known_bases)

  def check_version_constraints(self, min_v: str | None, max_v: str | None) -> str | None:
    """Validate whether the target framework version satisfies ``[min_v, max_v)``.

    Args:
        min_v: Optional inclusive minimum version string.
        max_v: Optional exclusive maximum version string.

    Returns:
        Warning string if constraints are violated, or ``None`` if satisfied/unknown.
    """
    if not min_v and not max_v:
      return None

    current: str | None = None
    target_fw = self.state.target_framework
    fw_conf = self.state.semantics.get_framework_config(target_fw)
    if fw_conf and "version" in fw_conf and isinstance(fw_conf["version"], str):
      current = fw_conf["version"]
    else:
      from ml_switcheroo.frameworks.base import get_adapter

      tgt_adapter = get_adapter(target_fw)
      pkg = (
        tgt_adapter.structural_traits.package_name
        if (tgt_adapter and tgt_adapter.structural_traits.package_name)
        else target_fw
      )
      try:
        current = importlib.metadata.version(pkg)
      except (importlib.metadata.PackageNotFoundError, ValueError, TypeError, RuntimeError):
        current = None

    if not current:
      return None

    def parse_v(v_str: str) -> tuple[int, ...]:
      """Split ``v_str`` into integer version segments.

      Args:
          v_str: Version string such as ``'1.2.3'``.

      Returns:
          Tuple of numeric version components.
      """
      return tuple(int(t) for t in re.split(r"[^\d]+", v_str) if t)

    curr_tuple = parse_v(current)

    if min_v and curr_tuple < parse_v(min_v):
      return f"Target {target_fw}@{current} is older than required {min_v}"

    if max_v and curr_tuple >= parse_v(max_v):
      return f"Target {target_fw}@{current} exceeds max supported {max_v}"

    return None

  def inject_argument_to_signature(
    self,
    node: cst.FunctionDef,
    arg_name: str,
    annotation: str | None,
  ) -> cst.FunctionDef:
    """Inject a parameter ``arg_name`` after ``self`` (or at index 0) if not already present.

    Args:
        node: Function definition CST node to update.
        arg_name: Parameter name to inject.
        annotation: Optional type annotation expression string.

    Returns:
        Modified ``cst.FunctionDef`` with the injected parameter.
    """
    params = list(node.params.params)
    insert_idx = 1 if (params and params[0].name.value == "self") else 0

    if any(p.name.value == arg_name for p in params):
      return node

    anno_node = cst.Annotation(annotation=self.create_dotted_name(annotation)) if annotation else None

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

    if params:
      params[-1] = params[-1].with_changes(comma=cst.MaybeSentinel.DEFAULT)

    return node.with_changes(params=node.params.with_changes(params=params))


__all__ = ["ApiPassState", "MappingResolver"]
