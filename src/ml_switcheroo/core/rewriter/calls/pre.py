"""Pre-processing Phase for Call Rewriting.

Handles functional unwrapping, plugin claims, and lifecycle method stripping.
Updated to remove dependencies on deleted legacy modules.
"""

import typing

import libcst as cst

from ml_switcheroo.core.hooks import get_hook
from ml_switcheroo.core.rewriter.calls.utils import is_functional_apply, log_diff, rewrite_stateful_call

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.interface import RewriterHost


@typing.runtime_checkable
class _TraitsObjectProtocol(typing.Protocol):
  """Protocol for traits objects exposing `functional_execution_method` and `implicit_method_roots`."""

  functional_execution_method: str | None
  implicit_method_roots: list[str]


@typing.runtime_checkable
class _SourceTraitsPropProtocol(typing.Protocol):
  """Protocol for rewriters exposing `source_traits` property."""

  @property
  def source_traits(self) -> _TraitsObjectProtocol:
    """Return source structural traits.

    Returns:
        Traits object.
    """
    ...


@typing.runtime_checkable
class _SourceTraitsMethodProtocol(typing.Protocol):
  """Protocol for rewriters exposing `_get_source_traits` method."""

  def _get_source_traits(self) -> _TraitsObjectProtocol:
    """Return source structural traits.

    Returns:
        Traits object.
    """
    ...


@typing.runtime_checkable
class _MappingGetterProtocol(typing.Protocol):
  """Protocol for rewriters providing `_get_mapping`."""

  def _get_mapping(self, name: str, silent: bool = False) -> object:
    """Look up target mapping for `name`.

    Args:
        name: API path string.
        silent: Suppress failure reporting.

    Returns:
        Target mapping or None.
    """
    ...


@typing.runtime_checkable
class _SourceLifecycleProtocol(typing.Protocol):
  """Protocol for rewriters providing `_get_source_lifecycle_lists`."""

  def _get_source_lifecycle_lists(self) -> tuple[set[str], set[str]]:
    """Return strip and warning lifecycle method sets.

    Returns:
        Tuple of `(strip_set, warn_set)`.
    """
    ...


@typing.runtime_checkable
class _WarningReporterProtocol(typing.Protocol):
  """Protocol for rewriters providing `_report_warning`."""

  def _report_warning(self, msg: str) -> None:
    """Report a warning message.

    Args:
        msg: Warning string.
    """
    ...


@typing.runtime_checkable
class _StatefulInspectorProtocol(typing.Protocol):
  """Protocol for rewriters providing `_is_stateful`."""

  def _is_stateful(self, name: str) -> bool:
    """Return True if `name` is marked stateful.

    Args:
        name: Variable identifier.

    Returns:
        Boolean flag.
    """
    ...


@typing.runtime_checkable
class _ModuleAliasCheckerProtocol(typing.Protocol):
  """Protocol for rewriters providing `_is_module_alias`."""

  def _is_module_alias(self, node: cst.CSTNode) -> bool:
    """Return True if `node` refers to an imported module alias.

    Args:
        node: CST node.

    Returns:
        Boolean flag.
    """
    ...


@typing.runtime_checkable
class _TargetTraitsGetterProtocol(typing.Protocol):
  """Protocol for rewriters providing `_get_target_traits`."""

  def _get_target_traits(self) -> object:
    """Return target structural traits.

    Returns:
        Traits object.
    """
    ...


class PreRewriterProtocol(typing.Protocol):
  """Minimal rewriter protocol required by handle_pre_checks and resolve_implicit_method."""

  @property
  def source_framework(self) -> str:
    """Return the source framework identifier.

    Returns:
        Source framework identifier string.
    """
    ...


def handle_pre_checks(
  rewriter: "RewriterHost | PreRewriterProtocol",
  original: cst.Call,
  updated: cst.Call,
  func_name: str | None,
) -> tuple[bool, cst.BaseExpression]:
  """Execute pre-lookup checks and transformations.

  Args:
      rewriter: The calling transformer (duck typing: needs _get_source_traits,
                _report_warning, _is_stateful, semantics, target_framework, ctx).
      original: The original CST node.
      updated: The updated CST node.
      func_name: Resolved function name.

  Returns:
      Tuple(handled, result_node).

  """
  # 1. Functional 'apply' unwrapping (Dynamic Trait)
  source_traits: _TraitsObjectProtocol | None = None
  if isinstance(rewriter, _SourceTraitsPropProtocol):
    source_traits = rewriter.source_traits
  elif isinstance(rewriter, _SourceTraitsMethodProtocol):
    source_traits = rewriter._get_source_traits()

  if source_traits is not None:
    unwrap_method = source_traits.functional_execution_method

    if is_functional_apply(original, unwrap_method):
      if isinstance(updated.func, cst.Attribute):
        receiver = updated.func.value
        # Strip the first argument (variables/params) and use receiver as callable
        # e.g. layer.apply(vars, x) -> layer(x)
        if len(updated.args) > 0:
          new_args = updated.args[1:]
        else:
          new_args = []

        result_node: cst.BaseExpression = updated.with_changes(func=receiver, args=new_args)
        log_diff("Functional Unwrap", original, result_node)
        return True, result_node

  # 2. Plugin Check (Explicit Requirement or ODL In-Place Metadata)
  plugin_claim = False
  is_inplace = False
  import operator

  try:
    ctx_obj: object = operator.attrgetter("context")(rewriter)
  except AttributeError:
    ctx_obj = None
  sem_obj: object = operator.attrgetter("semantics")(rewriter)

  if func_name:
    mapping = rewriter._get_mapping(func_name, silent=True) if isinstance(rewriter, _MappingGetterProtocol) else None

    if isinstance(mapping, dict) and "requires_plugin" in mapping:
      plugin_claim = True

    # Check ODL metadata for inplace flag
    get_def_fn = operator.attrgetter("get_definition")(sem_obj) if sem_obj is not None else None
    defn = get_def_fn(func_name) if callable(get_def_fn) else None
    if defn:
      _, details = defn
      if isinstance(details, dict) and details.get("is_inplace"):
        is_inplace = True

  # 2b. Heuristic: In-Place Unrolling
  should_unroll = False
  if is_inplace:
    should_unroll = True
  elif not plugin_claim and func_name and func_name.endswith("_") and not func_name.startswith("__"):
    should_unroll = True

  if should_unroll:
    hook = get_hook("unroll_inplace_ops")
    if hook:
      hook_ctx = operator.attrgetter("hook_context")(ctx_obj)
      new_node = hook(updated, hook_ctx)
      if new_node != updated:
        res_unroll: cst.BaseExpression = operator.itemgetter("v")({"v": new_node})
        log_diff("In-place Unroll", updated, res_unroll)
        return True, res_unroll

  # 3. Lifecycle Method Handling (Strip/Warn)
  if isinstance(rewriter, _SourceLifecycleProtocol):
    strip_set, warn_set = rewriter._get_source_lifecycle_lists()

    if not plugin_claim and isinstance(original.func, cst.Attribute) and isinstance(original.func.attr, cst.Name):
      method_name = original.func.attr.value

      if method_name in strip_set and isinstance(updated.func, cst.Attribute):
        assert isinstance(rewriter, _WarningReporterProtocol)
        rewriter._report_warning(f"Stripped framework-specific lifecycle method '.{method_name}()'.")
        result_node = updated.func.value
        log_diff("Lifecycle Strip", original, result_node)
        return True, result_node

      if method_name in warn_set and isinstance(updated.func, cst.Attribute):
        assert isinstance(rewriter, _WarningReporterProtocol)
        rewriter._report_warning(f"Ignored model state method '.{method_name}()'.")
        result_node = updated.func.value
        log_diff("Lifecycle Warn", original, result_node)
        return True, result_node

  # 4. Stateful Call
  if func_name and isinstance(rewriter, _StatefulInspectorProtocol) and rewriter._is_stateful(func_name):
    tgt_fw = str(operator.attrgetter("target_framework")(rewriter))
    get_fw_cfg_fn = operator.attrgetter("get_framework_config")(sem_obj) if sem_obj is not None else None
    fw_config = get_fw_cfg_fn(tgt_fw) if callable(get_fw_cfg_fn) else None
    raw_stateful = fw_config.get("stateful_call") if isinstance(fw_config, dict) else None
    if isinstance(raw_stateful, dict) and raw_stateful:
      stateful_spec: dict[str, str] = {str(k): str(v) for k, v in raw_stateful.items()}
      result_node = rewrite_stateful_call(rewriter, updated, func_name, stateful_spec)
      log_diff("State Mechanism", original, result_node)
      return True, result_node

  return False, updated


def resolve_implicit_method(
  rewriter: "RewriterHost | PreRewriterProtocol",
  original: cst.Call,
  func_name: str | None,
) -> str | None:
  """Attempt to resolve method calls on objects to full API paths.

  Args:
      rewriter: The calling transformer object.
      original: The original Call node from libcst.
      func_name: The optional function name string.

  Returns:
      The resolved full API path if found, otherwise None.

  """
  if isinstance(original.func, cst.Attribute) and isinstance(original.func.attr, cst.Name):
    receiver = original.func.value
    leaf_method = original.func.attr.value

    # Check for 'self'
    is_self = isinstance(receiver, cst.Name) and receiver.value == "self"

    # Check if module alias (requires rewriter alias checker)
    is_module = rewriter._is_module_alias(receiver) if isinstance(rewriter, _ModuleAliasCheckerProtocol) else False

    if not is_self and not is_module:
      import operator

      try:
        ctx_obj: object = operator.attrgetter("context")(rewriter)
      except AttributeError:
        ctx_obj = None
      # --- 1. Symbol Table Inference ---
      sym_table = operator.attrgetter("symbol_table")(ctx_obj) if ctx_obj is not None else None
      if sym_table is not None:
        sym_type = sym_table.get_type(receiver)
        if sym_type is not None:
          sym_name = str(operator.attrgetter("name")(sym_type))
          candidate_api = f"{sym_name}.{leaf_method}"

          try:
            sym_fw: object = operator.attrgetter("framework")(sym_type)
          except AttributeError:
            sym_fw = None
          if "Tensor" in sym_name and sym_fw:
            candidate_api = f"{sym_fw}.Tensor.{leaf_method}"

          if isinstance(rewriter, _MappingGetterProtocol):
            mapping = rewriter._get_mapping(candidate_api, silent=True)
            if mapping:
              return candidate_api

      # --- 2. Legacy Heuristic Fallback ---
      r_obj: object = rewriter
      if isinstance(r_obj, _TargetTraitsGetterProtocol):
        if isinstance(r_obj, _SourceTraitsPropProtocol):
          traits = r_obj.source_traits
        else:
          sem_obj: object = operator.attrgetter("semantics")(rewriter)
          get_fw_cfg = operator.attrgetter("get_framework_config")(sem_obj) if sem_obj is not None else None
          config_dict = get_fw_cfg(rewriter.source_framework) if callable(get_fw_cfg) else {}
          from ml_switcheroo.semantics.schema import StructuralTraits

          raw_traits = config_dict.get("traits", {}) if isinstance(config_dict, dict) else {}
          traits = StructuralTraits.model_validate(raw_traits if isinstance(raw_traits, dict) else {})

        implicit_roots = traits.implicit_method_roots

        for root in implicit_roots:
          candidate_api = f"{root}.{leaf_method}"
          if isinstance(rewriter, _MappingGetterProtocol):
            mapping = rewriter._get_mapping(candidate_api, silent=True)
            if mapping:
              return candidate_api

  return None
