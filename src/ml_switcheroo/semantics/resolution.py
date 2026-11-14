"""Multi-tier semantic operation resolution and variant dispatch logic."""

from __future__ import annotations

import types
import typing

import pydantic

from ml_switcheroo.core.dsl import FrameworkVariant, OperationDef
from ml_switcheroo.exceptions import SemanticsResolutionError
from ml_switcheroo.semantics.schema import FrameworkAlias, FrameworkTraits, PluginTraits, StructuralTraits

if typing.TYPE_CHECKING:
  from ml_switcheroo.semantics.manager import SemanticsManager
  from ml_switcheroo.semantics.schema import OperationDataModel


@typing.runtime_checkable
class _DefinitionByIdProtocol(typing.Protocol):
  """Protocol for legacy/mock semantics objects providing ``get_definition_by_id``."""

  def get_definition_by_id(self, abstract_id: str) -> dict[str, pydantic.JsonValue] | OperationDef | None:
    """Return raw operation dictionary or ``OperationDef`` by abstract ID.

    Args:
        abstract_id: Canonical abstract operation identifier.

    Returns:
        Raw operation dictionary, ``OperationDef``, or ``None``.
    """


@typing.runtime_checkable
class _LegacyGetDefinitionProtocol(typing.Protocol):
  """Protocol for legacy/mock semantics objects providing ``get_definition``."""

  def get_definition(self, api_name: str) -> tuple[str, dict[str, pydantic.JsonValue]] | None:
    """Resolve concrete API path to ``(abstract_id, details)``.

    Args:
        api_name: Concrete API path or abstract operation ID.

    Returns:
        Tuple ``(abstract_id, details)`` or ``None``.
    """


@typing.runtime_checkable
class _LegacyResolveVariantProtocol(typing.Protocol):
  """Protocol for legacy/mock semantics objects providing ``resolve_variant``."""

  def resolve_variant(
    self, abstract_id: str, target_framework: str
  ) -> dict[str, pydantic.JsonValue] | FrameworkVariant | None:
    """Resolve raw variant dictionary or ``FrameworkVariant``.

    Args:
        abstract_id: Canonical abstract operation identifier.
        target_framework: Target framework key.

    Returns:
        Raw variant dictionary, ``FrameworkVariant``, or ``None``.
    """


@typing.runtime_checkable
class _DuckPluginTraitsProtocol(typing.Protocol):
  """Protocol for custom test plugin-traits objects exposing ``has_numpy_compatible_arrays``.

  Attributes:
      has_numpy_compatible_arrays: Whether the framework supports NumPy-compatible array methods.
  """

  has_numpy_compatible_arrays: bool


@typing.runtime_checkable
class _LegacyFrameworkConfigProtocol(typing.Protocol):
  """Protocol for legacy/mock semantics objects providing ``get_framework_config``."""

  def get_framework_config(
    self, framework: str
  ) -> (
    typing.Mapping[str, pydantic.JsonValue | StructuralTraits | PluginTraits | _DuckPluginTraitsProtocol | FrameworkAlias]
    | FrameworkTraits
    | None
  ):
    """Return raw framework configuration dictionary or ``FrameworkTraits``.

    Args:
        framework: Target framework key.

    Returns:
        Raw framework configuration dictionary, ``FrameworkTraits``, or ``None``.
    """


class ResolutionContext(pydantic.BaseModel):
  """Context object encapsulating source and target frameworks during semantic resolution.

  Attributes:
      source_framework: The source framework (`str`).
      target_framework: The target framework (`str`).
      current_scope: The current scope (`str`).
  """

  model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

  source_framework: str
  target_framework: str
  current_scope: str = "global"

  def __init__(
    self,
    source_framework: str,
    target_framework: str,
    current_scope: str = "global",
  ) -> None:
    """Initialize ResolutionContext.

    Args:
        source_framework: Source framework key.
        target_framework: Target framework key.
        current_scope: Current semantic resolution scope.
    """
    super().__init__(
      source_framework=source_framework,
      target_framework=target_framework,
      current_scope=current_scope,
    )


class SemanticResolver:
  """Lightweight resolver mapping framework API calls to abstract operations and target APIs."""

  def __init__(
    self,
    operation_maps: typing.Mapping[str, OperationDef | OperationDataModel | dict[str, pydantic.JsonValue]],
  ) -> None:
    """Initialize the resolver with an operation mapping dictionary.

    Args:
        operation_maps: Dictionary mapping abstract operation names to their variant definitions.
    """
    self.operation_maps = operation_maps

  def resolve(self, framework: str, api_call: str) -> str | None:
    """Resolve a concrete framework API call string to its abstract operation name.

    Args:
        framework: Source framework key (e.g., 'torch').
        api_call: Concrete API string (e.g., 'torch.add').

    Returns:
        Matching abstract operation identifier if found, else None.

    Raises:
        SemanticsResolutionError: If an operation's ``variants`` field is not a dictionary.
    """
    for abstract_op, mapping in self.operation_maps.items():
      if isinstance(mapping, OperationDef):
        typed_var = mapping.variants.get(framework)
        if typed_var is not None and typed_var.api == api_call:
          return abstract_op
        continue
      raw_map = mapping.model_dump() if isinstance(mapping, pydantic.BaseModel) else mapping
      variants = raw_map.get("variants") or {}
      if not isinstance(variants, dict):
        raise SemanticsResolutionError(f"Expected dict for variants of '{abstract_op}', got {type(variants).__name__}.")
      fw_variant = variants.get(framework)
      if isinstance(fw_variant, dict) and fw_variant.get("api") == api_call:
        return abstract_op
    return None

  def dispatch(self, abstract_op: str, target_framework: str) -> str | None:
    """Dispatch an abstract operation name to the target framework's concrete API path.

    Args:
        abstract_op: Abstract operation identifier (e.g., 'Add').
        target_framework: Target framework key (e.g., 'jax').

    Returns:
        Target concrete API path if defined, else None.

    Raises:
        SemanticsResolutionError: If ``abstract_op`` has a non-dictionary ``variants`` field.
    """
    mapping = self.operation_maps.get(abstract_op)
    if mapping is not None:
      if isinstance(mapping, OperationDef):
        typed_var = mapping.variants.get(target_framework)
        return typed_var.api if typed_var is not None else None
      raw_map = mapping.model_dump() if isinstance(mapping, pydantic.BaseModel) else mapping
      variants = raw_map.get("variants") or {}
      if not isinstance(variants, dict):
        raise SemanticsResolutionError(f"Expected dict for variants of '{abstract_op}', got {type(variants).__name__}.")
      fw_variant = variants.get(target_framework)
      if isinstance(fw_variant, dict):
        api = fw_variant.get("api")
        return str(api) if api is not None else None
    return None


class ResolutionEngine:
  """Modular resolution engine handling inheritance-aware variant lookup and reverse-index queries."""

  @staticmethod
  def resolve_variant(
    manager: SemanticsManager, abstract_id: str, target_framework: str
  ) -> dict[str, pydantic.JsonValue] | None:
    """Resolve the implementation variant dictionary for ``abstract_id`` in ``target_framework``.

    Traverses framework inheritance chains (e.g., ``flax_nnx -> jax``) and case-insensitive
    operation aliases, caching results in ``manager._variant_cache``.

    Args:
        manager: SemanticsManager instance holding operation data and inheritance links.
        abstract_id: Abstract operation identifier to resolve.
        target_framework: Target framework key.

    Returns:
        Variant mapping dictionary if resolved, else None.
    """
    cache_key = (abstract_id, target_framework)
    try:
      _ = manager._variant_cache
    except AttributeError:
      manager._variant_cache = {}
    if cache_key in manager._variant_cache:
      return manager._variant_cache[cache_key]

    defn = manager.data.get(abstract_id)
    if not defn and isinstance(abstract_id, str):
      defn = manager.data.get(abstract_id.lower()) or manager.data.get(abstract_id.capitalize())
    if not defn:
      manager._variant_cache[cache_key] = None
      return None
    raw_variants = defn.get("variants")
    variants: dict[str, dict[str, pydantic.JsonValue] | None] = (
      {str(k): (v if isinstance(v, dict) else None) for k, v in raw_variants.items()}
      if isinstance(raw_variants, dict)
      else {}
    )
    if target_framework in variants:
      result = variants[target_framework]
      manager._variant_cache[cache_key] = result
      return result

    curr = target_framework
    limit = 5
    while limit > 0:
      parent = manager._resolve_inheritance(curr)
      if not parent:
        break
      if parent in variants:
        result = variants[parent]
        manager._variant_cache[cache_key] = result
        return result
      curr = parent
      limit -= 1

    for candidate_id in [abstract_id.lower(), abstract_id.capitalize()]:
      if candidate_id != abstract_id:
        alt_defn = manager.data.get(candidate_id)
        if alt_defn:
          alt_vars = alt_defn.get("variants")
          if isinstance(alt_vars, dict) and target_framework in alt_vars:
            alt_res = alt_vars[target_framework]
            res_dict: dict[str, pydantic.JsonValue] | None = alt_res if isinstance(alt_res, dict) else None
            manager._variant_cache[cache_key] = res_dict
            return res_dict

    manager._variant_cache[cache_key] = None
    return None

  @staticmethod
  def get_definition(manager: SemanticsManager, api_name: str) -> tuple[str, dict[str, pydantic.JsonValue]] | None:
    """Look up the `(abstract_id, details)` tuple for a concrete API name or abstract key.

    Args:
        manager: SemanticsManager instance holding `_reverse_index` and `data`.
        api_name: Concrete API path (e.g., 'torch.abs') or abstract operation ID.

    Returns:
        Tuple of `(abstract_id, operation_details_dict)` if found, else None.
    """
    try:
      _ = manager._reverse_index
    except AttributeError:
      manager._reverse_index = {}
    res: tuple[str, dict[str, pydantic.JsonValue]] | None = manager._reverse_index.get(api_name)
    if res:
      return res

    lower_name = api_name.lower()
    cap_name = api_name.capitalize()
    for candidate in [api_name, lower_name, cap_name]:
      if candidate in manager.data and manager.data[candidate].get("variants"):
        return (candidate, manager.data[candidate])

    for candidate in [api_name, lower_name, cap_name]:
      if candidate in manager.data:
        return (candidate, manager.data[candidate])

    return None

  @staticmethod
  def _coerce_operation_def(op_name: str, raw_dict: typing.Mapping[str, pydantic.JsonValue]) -> OperationDef | None:
    """Validate a raw operation dictionary into an ``OperationDef`` on demand.

    Args:
        op_name: Canonical operation identifier fallback.
        raw_dict: Raw operation mapping.

    Returns:
        Validated ``OperationDef`` or ``None`` if validation fails.
    """
    payload = dict(raw_dict)
    payload.setdefault("operation", op_name)
    payload.setdefault("variants", {})
    try:
      return OperationDef.model_validate(payload)
    except pydantic.ValidationError:
      return None

  @staticmethod
  def _ensure_typed_stores(
    manager: SemanticsManager,
  ) -> tuple[
    dict[str, OperationDef],
    dict[str, tuple[str, OperationDef]],
    dict[tuple[str, str], FrameworkVariant | None],
    dict[str, FrameworkTraits],
  ]:
    """Return typed store dictionaries on ``manager``, initializing missing attributes on test subclasses.

    Args:
        manager: ``SemanticsManager`` instance or test subclass.

    Returns:
        4-tuple ``(operations, _typed_reverse_index, _typed_variant_cache, framework_traits)``.
    """
    try:
      ops = manager.operations
    except AttributeError:
      ops = {}
      manager.operations = ops

    try:
      rev = manager._typed_reverse_index
    except AttributeError:
      rev = {}
      manager._typed_reverse_index = rev

    try:
      vcache = manager._typed_variant_cache
    except AttributeError:
      vcache = {}
      manager._typed_variant_cache = vcache

    try:
      ftraits = manager.framework_traits
    except AttributeError:
      ftraits = {}
      manager.framework_traits = ftraits

    return ops, rev, vcache, ftraits

  @classmethod
  def get_operation(cls, manager: SemanticsManager, op_name: str) -> OperationDef | None:
    """Look up a validated ``OperationDef`` by abstract operation ID or concrete API path.

    Args:
        manager: ``SemanticsManager`` instance holding typed ``operations``.
        op_name: Canonical abstract operation ID or concrete API path.

    Returns:
        Validated ``OperationDef`` instance if resolved, else ``None``.
    """
    from ml_switcheroo.semantics.manager import SemanticsManager as _MgrCls

    by_id_method = manager.get_definition_by_id
    if not (isinstance(by_id_method, types.MethodType) and by_id_method.__func__ is _MgrCls.get_definition_by_id):
      overridden = manager.get_definition_by_id(op_name)
      if isinstance(overridden, dict):
        return cls._coerce_operation_def(op_name, overridden)
      return None

    ops, typed_rev, _, _ = cls._ensure_typed_stores(manager)
    if op_name in ops:
      return ops[op_name]

    rev = typed_rev.get(op_name)
    if rev is not None:
      return rev[1]

    for candidate in (op_name.lower(), op_name.capitalize()):
      if candidate in ops:
        return ops[candidate]

    raw_entry = manager.data.get(op_name)
    if raw_entry is None:
      raw_def = cls.get_definition(manager, op_name)
      if raw_def is not None:
        resolved_id, raw_entry = raw_def
        coerced = cls._coerce_operation_def(resolved_id, raw_entry)
        if coerced is not None:
          ops[resolved_id] = coerced
        return coerced
      return None

    coerced = cls._coerce_operation_def(op_name, raw_entry)
    if coerced is not None:
      ops[op_name] = coerced
    return coerced

  @classmethod
  def resolve_operation(cls, manager: SemanticsManager, api_name: str) -> tuple[str, OperationDef] | None:
    """Resolve a concrete API path or abstract ID to ``(abstract_id, OperationDef)``.

    Args:
        manager: ``SemanticsManager`` instance.
        api_name: Concrete API path (e.g., ``'torch.abs'``) or abstract operation ID.

    Returns:
        Tuple ``(abstract_id, OperationDef)`` if found, else ``None``.
    """
    from ml_switcheroo.semantics.manager import SemanticsManager as _MgrCls

    get_def_method = manager.get_definition
    if not (isinstance(get_def_method, types.MethodType) and get_def_method.__func__ is _MgrCls.get_definition):
      overridden = manager.get_definition(api_name)
      if overridden is None:
        return None
      abs_id, raw_details = overridden
      coerced = cls._coerce_operation_def(abs_id, raw_details)
      return (abs_id, coerced) if coerced is not None else None

    ops, typed_rev, _, _ = cls._ensure_typed_stores(manager)
    rev = typed_rev.get(api_name)
    if rev is not None:
      return rev

    for candidate in (api_name, api_name.lower(), api_name.capitalize()):
      if candidate in ops and ops[candidate].variants:
        return (candidate, ops[candidate])
    for candidate in (api_name, api_name.lower(), api_name.capitalize()):
      if candidate in ops:
        return (candidate, ops[candidate])

    raw_res = cls.get_definition(manager, api_name)
    if raw_res is None:
      return None
    abs_id, raw_details = raw_res
    coerced = cls._coerce_operation_def(abs_id, raw_details)
    if coerced is not None:
      ops[abs_id] = coerced
      return (abs_id, coerced)
    return None

  @classmethod
  def _lookup_variant_in_op(
    cls,
    manager: SemanticsManager,
    op_def: OperationDef,
    target_framework: str,
  ) -> tuple[bool, FrameworkVariant | None]:
    """Search ``op_def.variants`` across ``target_framework`` and its inheritance ancestors.

    Args:
        manager: ``SemanticsManager`` providing ``_resolve_inheritance``.
        op_def: Validated ``OperationDef`` to inspect.
        target_framework: Target framework identifier.

    Returns:
        ``(found, variant_or_none)`` distinguishing explicit ``None`` variants from missing keys.
    """
    if target_framework in op_def.variants:
      return True, op_def.variants[target_framework]

    curr = target_framework
    for _ in range(5):
      parent = manager._resolve_inheritance(curr)
      if not parent:
        break
      if parent in op_def.variants:
        return True, op_def.variants[parent]
      curr = parent
    return False, None

  @classmethod
  def get_variant(cls, manager: SemanticsManager, op_name: str, target_framework: str) -> FrameworkVariant | None:
    """Resolve the typed ``FrameworkVariant`` for ``op_name`` in ``target_framework``.

    Args:
        manager: ``SemanticsManager`` instance holding typed ``operations`` and caches.
        op_name: Canonical abstract operation identifier.
        target_framework: Target framework key.

    Returns:
        Validated ``FrameworkVariant`` if mapped, else ``None``.
    """
    from ml_switcheroo.semantics.manager import SemanticsManager as _MgrCls

    rv_method = manager.resolve_variant
    if not (isinstance(rv_method, types.MethodType) and rv_method.__func__ is _MgrCls.resolve_variant):
      raw_override = manager.resolve_variant(op_name, target_framework)
      if not isinstance(raw_override, dict):
        return None
      try:
        return FrameworkVariant.model_validate(raw_override)
      except pydantic.ValidationError:
        return None

    ops, _, typed_vcache, _ = cls._ensure_typed_stores(manager)
    cache_key = (op_name, target_framework)
    if cache_key in typed_vcache:
      return typed_vcache[cache_key]

    op_def = ops.get(op_name) or ops.get(op_name.lower()) or ops.get(op_name.capitalize())
    if op_def is not None:
      found, variant = cls._lookup_variant_in_op(manager, op_def, target_framework)
      if found:
        typed_vcache[cache_key] = variant
        return variant
      for alt_id in (op_name.lower(), op_name.capitalize()):
        alt_op = ops.get(alt_id)
        if alt_id != op_name and alt_op is not None and target_framework in alt_op.variants:
          alt_var = alt_op.variants[target_framework]
          typed_vcache[cache_key] = alt_var
          return alt_var

    raw_fallback = cls.resolve_variant(manager, op_name, target_framework)
    if isinstance(raw_fallback, dict):
      try:
        validated = FrameworkVariant.model_validate(raw_fallback)
        typed_vcache[cache_key] = validated
        return validated
      except pydantic.ValidationError:
        return None

    typed_vcache[cache_key] = None
    return None

  @staticmethod
  def _coerce_framework_traits(
    raw_conf: typing.Mapping[
      str, pydantic.JsonValue | StructuralTraits | PluginTraits | _DuckPluginTraitsProtocol | FrameworkAlias
    ],
  ) -> FrameworkTraits:
    """Coerce a raw framework configuration mapping into ``FrameworkTraits``.

    Args:
        raw_conf: Framework configuration dictionary.

    Returns:
        Validated ``FrameworkTraits`` model.
    """
    if not raw_conf:
      return FrameworkTraits()
    try:
      return FrameworkTraits.model_validate(raw_conf)
    except pydantic.ValidationError:
      raw_struct = raw_conf.get("traits")
      struct_traits = (
        raw_struct
        if isinstance(raw_struct, StructuralTraits)
        else (StructuralTraits.model_validate(raw_struct) if isinstance(raw_struct, dict) else StructuralTraits())
      )
      raw_plugin = raw_conf.get("plugin_traits")
      plug_traits = (
        raw_plugin
        if isinstance(raw_plugin, PluginTraits)
        else (
          PluginTraits.model_validate(raw_plugin)
          if isinstance(raw_plugin, dict)
          else (
            PluginTraits(has_numpy_compatible_arrays=bool(raw_plugin.has_numpy_compatible_arrays))
            if isinstance(raw_plugin, _DuckPluginTraitsProtocol)
            else PluginTraits()
          )
        )
      )
      raw_alias = raw_conf.get("alias")
      alias_obj = (
        raw_alias
        if isinstance(raw_alias, FrameworkAlias)
        else (
          FrameworkAlias.model_validate(raw_alias)
          if isinstance(raw_alias, dict) and "module" in raw_alias and "name" in raw_alias
          else None
        )
      )
      raw_stateful = raw_conf.get("stateful_call")
      stateful_map = {str(k): str(v) for k, v in raw_stateful.items()} if isinstance(raw_stateful, dict) else None
      raw_extends = raw_conf.get("extends")
      return FrameworkTraits(
        extends=str(raw_extends) if isinstance(raw_extends, str) else None,
        alias=alias_obj,
        stateful_call=stateful_map,
        traits=struct_traits,
        plugin_traits=plug_traits,
      )

  @classmethod
  def get_framework_traits(cls, manager: SemanticsManager, framework: str) -> FrameworkTraits:
    """Return the validated ``FrameworkTraits`` model for ``framework``.

    Args:
        manager: ``SemanticsManager`` instance holding ``framework_traits``.
        framework: Framework identifier key.

    Returns:
        Validated ``FrameworkTraits`` (empty default model if unknown).
    """
    from ml_switcheroo.semantics.manager import SemanticsManager as _MgrCls

    cfg_method = manager.get_framework_config
    if not (isinstance(cfg_method, types.MethodType) and cfg_method.__func__ is _MgrCls.get_framework_config):
      raw_override = manager.get_framework_config(framework)
      return cls._coerce_framework_traits(raw_override)

    _, _, _, ftraits = cls._ensure_typed_stores(manager)
    stored = ftraits.get(framework)
    try:
      raw_conf = manager.framework_configs.get(framework)
    except AttributeError:
      raw_conf = None
    if raw_conf is not None:
      coerced = cls._coerce_framework_traits(raw_conf)
      ftraits[framework] = coerced
      return coerced
    return stored if stored is not None else FrameworkTraits()


def _is_real_semantics_manager(
  obj: SemanticsManager
  | _DefinitionByIdProtocol
  | _LegacyGetDefinitionProtocol
  | _LegacyResolveVariantProtocol
  | _LegacyFrameworkConfigProtocol,
) -> typing.TypeGuard[SemanticsManager]:
  """Check whether ``obj`` is a genuine ``SemanticsManager`` (excluding ``MagicMock(spec=SemanticsManager)``).

  Args:
      obj: Candidate semantics object.

  Returns:
      ``True`` if ``type(obj)`` is ``SemanticsManager`` or a concrete subclass thereof.
  """
  from ml_switcheroo.semantics.manager import SemanticsManager as _MgrCls

  return issubclass(type(obj), _MgrCls)


def query_operation(
  semantics: SemanticsManager | _DefinitionByIdProtocol | None,
  op_name: str,
) -> OperationDef | None:
  """Polymorphically query a typed ``OperationDef`` from ``SemanticsManager`` or a test protocol stub.

  Args:
      semantics: ``SemanticsManager`` instance or lightweight test double.
      op_name: Canonical abstract operation identifier.

  Returns:
      Validated ``OperationDef`` if resolved, else ``None``.
  """
  if semantics is None:
    return None
  if _is_real_semantics_manager(semantics):
    return semantics.get_operation(op_name)
  try:
    raw = semantics.get_definition_by_id(op_name)
  except AttributeError:
    return None
  if isinstance(raw, OperationDef):
    return raw
  if isinstance(raw, dict):
    return ResolutionEngine._coerce_operation_def(op_name, raw)
  return None


def query_resolve_operation(
  semantics: SemanticsManager | _LegacyGetDefinitionProtocol | None,
  api_name: str,
) -> tuple[str, OperationDef] | None:
  """Resolve ``api_name`` to ``(abstract_id, OperationDef)`` across ``SemanticsManager`` and test mocks.

  Args:
      semantics: ``SemanticsManager`` instance or mock test double.
      api_name: Concrete API path or abstract operation identifier.

  Returns:
      Tuple ``(abstract_id, OperationDef)`` if resolved, else ``None``.
  """
  if semantics is None:
    return None
  if _is_real_semantics_manager(semantics):
    return semantics.resolve_operation(api_name)
  try:
    raw_pair = semantics.get_definition(api_name)
  except AttributeError:
    return None
  if isinstance(raw_pair, tuple):
    abs_id, raw_details = raw_pair
    coerced = ResolutionEngine._coerce_operation_def(abs_id, raw_details)
    if coerced is not None:
      return (abs_id, coerced)
  return None


def query_variant(
  semantics: SemanticsManager | _LegacyResolveVariantProtocol | None,
  op_name: str,
  framework: str,
) -> FrameworkVariant | None:
  """Polymorphically query a typed ``FrameworkVariant`` from ``SemanticsManager`` or a test protocol stub.

  Args:
      semantics: ``SemanticsManager`` instance or lightweight test double.
      op_name: Canonical abstract operation identifier.
      framework: Target framework key.

  Returns:
      Validated ``FrameworkVariant`` if resolved, else ``None``.
  """
  if semantics is None:
    return None
  if _is_real_semantics_manager(semantics):
    return semantics.get_variant(op_name, framework)
  try:
    raw = semantics.resolve_variant(op_name, framework)
  except AttributeError:
    return None
  if isinstance(raw, FrameworkVariant):
    return raw
  if isinstance(raw, dict):
    try:
      return FrameworkVariant.model_validate(raw)
    except pydantic.ValidationError:
      return None
  return None


def query_framework_traits(
  semantics: SemanticsManager | _LegacyFrameworkConfigProtocol | None,
  framework: str,
) -> FrameworkTraits:
  """Polymorphically query typed ``FrameworkTraits`` from ``SemanticsManager`` or a test protocol stub.

  Args:
      semantics: ``SemanticsManager`` instance or lightweight test double.
      framework: Framework identifier key.

  Returns:
      Validated ``FrameworkTraits`` instance.
  """
  if semantics is None:
    return FrameworkTraits()
  if _is_real_semantics_manager(semantics):
    return semantics.get_framework_traits(framework)
  try:
    raw = semantics.get_framework_config(framework)
  except AttributeError:
    return FrameworkTraits()
  if isinstance(raw, FrameworkTraits):
    return raw
  if isinstance(raw, dict):
    return ResolutionEngine._coerce_framework_traits(raw)
  return FrameworkTraits()
