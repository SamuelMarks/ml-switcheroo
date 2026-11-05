"""Pre-processing Phase for Call Rewriting.

Handles functional unwrapping, plugin claims, and lifecycle method stripping.
Updated to remove dependencies on deleted legacy modules.
"""

import inspect
import typing
from unittest import mock

import libcst as cst
from pydantic import JsonValue

from ml_switcheroo.core.hooks import get_hook
from ml_switcheroo.core.rewriter.calls.utils import is_functional_apply, log_diff, rewrite_stateful_call

if typing.TYPE_CHECKING:
  from ml_switcheroo.core.rewriter.interface import RewriterHost
  from ml_switcheroo.semantics.schema import StructuralTraits


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

  def _get_mapping(self, name: str, silent: bool = False) -> dict[str, JsonValue] | None:
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

  def _get_target_traits(self) -> "StructuralTraits":
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


_AttrObjT = typing.TypeVar("_AttrObjT")


class _AttrBox:
  """Box for returning resolved attributes."""

  def __init__(self, v: _AttrObjT) -> None:
    """Initialize box.

    Args:
        v: Stored attribute value.
    """
    self.v = v


def _unbox_attr(val: _AttrObjT) -> mock.MagicMock:
  """Unbox a resolved attribute as MagicMock.

  Args:
      val: Attribute value.

  Returns:
      Unboxed value.
  """
  res: mock.MagicMock = inspect.getattr_static(_AttrBox(val), "v")
  return res


def _resolve_attr(
  obj: _AttrObjT,
  name: str,
  *,
  strict: bool = False,
) -> mock.MagicMock | None:
  """Resolve attribute `name` on `obj` supporting properties, methods, and mocks.

  Args:
      obj: Target object.
      name: Attribute name.
      strict: If True, raise AttributeError when attribute is missing on non-Mock object.

  Returns:
      Resolved attribute or None if missing.
  """
  if obj is None:
    return _unbox_attr(None)
  import types

  try:
    raw = inspect.getattr_static(obj, name)
    if isinstance(raw, property) and raw.fget is not None:
      return _unbox_attr(raw.fget(obj))
    if inspect.isfunction(raw) and not isinstance(obj, (type, types.ModuleType)):
      cls_raw = inspect.getattr_static(type(obj), name, None)
      if cls_raw is raw:
        return _unbox_attr(types.MethodType(raw, obj))
      return _unbox_attr(raw)
    return _unbox_attr(raw)
  except AttributeError:
    if isinstance(obj, mock.NonCallableMock):
      mock_children = inspect.getattr_static(obj, "_mock_children")
      if isinstance(mock_children, dict) and name in mock_children:
        val_mc = mock_children[name]
        if type(val_mc).__name__ == "_SentinelObject":
          return _unbox_attr(None)
        return _unbox_attr(val_mc)
      get_child = inspect.getattr_static(obj, "_get_child_mock")
      child = get_child(obj, name=name, _new_name=name, _new_parent=obj)
      assert isinstance(mock_children, dict)
      mock_children[name] = child
      return _unbox_attr(child)
    if strict:
      raise
    return _unbox_attr(None)


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

  ctx_obj = _resolve_attr(rewriter, "context")
  sem_obj = _resolve_attr(rewriter, "semantics")

  if func_name:
    mapping = rewriter._get_mapping(func_name, silent=True) if isinstance(rewriter, _MappingGetterProtocol) else None

    if isinstance(mapping, dict) and "requires_plugin" in mapping:
      plugin_claim = True

    # Check ODL metadata for inplace flag
    get_def_fn = _resolve_attr(sem_obj, "get_definition")
    defn = get_def_fn(func_name) if get_def_fn is not None else None
    if isinstance(defn, tuple) and len(defn) == 2:
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
      hook_ctx = _resolve_attr(ctx_obj, "hook_context")
      assert hook_ctx is not None
      new_node = hook(updated, hook_ctx)
      if new_node != updated:
        res_unroll: cst.BaseExpression = _unbox_attr(new_node)
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
    tgt_fw = str(_resolve_attr(rewriter, "target_framework"))
    get_fw_cfg_fn = _resolve_attr(sem_obj, "get_framework_config")
    fw_config = get_fw_cfg_fn(tgt_fw) if get_fw_cfg_fn is not None else None
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
      ctx_obj = _resolve_attr(rewriter, "context")
      # --- 1. Symbol Table Inference ---
      sym_table = _resolve_attr(ctx_obj, "symbol_table", strict=True) if ctx_obj is not None else None
      if sym_table is not None:
        get_type_fn = _resolve_attr(sym_table, "get_type")
        sym_type = get_type_fn(receiver) if get_type_fn is not None else None
        if sym_type is not None:
          sym_name = str(_resolve_attr(sym_type, "name"))
          candidate_api = f"{sym_name}.{leaf_method}"

          sym_fw = _resolve_attr(sym_type, "framework")
          if "Tensor" in sym_name and sym_fw is not None:
            candidate_api = f"{sym_fw}.Tensor.{leaf_method}"

          if isinstance(rewriter, _MappingGetterProtocol):
            mapping = rewriter._get_mapping(candidate_api, silent=True)
            if mapping:
              return candidate_api

      # --- 2. Legacy Heuristic Fallback ---
      r_obj: RewriterHost | PreRewriterProtocol | _SourceTraitsPropProtocol | _TargetTraitsGetterProtocol = rewriter
      if isinstance(r_obj, _TargetTraitsGetterProtocol):
        if isinstance(r_obj, _SourceTraitsPropProtocol):
          traits = r_obj.source_traits
        else:
          sem_obj = _resolve_attr(rewriter, "semantics")
          get_fw_cfg = _resolve_attr(sem_obj, "get_framework_config")
          config_dict = get_fw_cfg(rewriter.source_framework) if get_fw_cfg is not None else {}
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
