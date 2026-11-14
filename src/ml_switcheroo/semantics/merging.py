"""Merging Logic for Semantic Knowledge Base.

This module handles the aggregation of disparate knowledge sources (Specs, Snapshots)
into the central SemanticsManager. It resolves conflicts based on Tier Precedence
and merging framework-specific configurations.

Handles:
- Merging Specification Tiers (Math, Neural, Extras).
- Merging Snapshot Overlays (Framework mappings).
- Conflict resolution and prioritization (Warning vs Silencing).
- Merging Framework Traits and Import Maps.
- Merging Patterns.
"""

import logging
import typing
import warnings
from dataclasses import dataclass

from ml_switcheroo_ir.schema.ghost import SemanticTier
from pydantic import BaseModel, JsonValue, ValidationError

from ml_switcheroo.core.dsl import FrameworkVariant, OperationDef, PatternDef
from ml_switcheroo.enums import RuntimeTensorOrValue
from ml_switcheroo.exceptions import SemanticLoadError
from ml_switcheroo.semantics.schema import FrameworkTraits

JsonValueLike: typing.TypeAlias = (
  bool
  | int
  | float
  | str
  | OperationDef
  | PatternDef
  | BaseModel
  | typing.Sequence["JsonValueLike"]
  | typing.Collection["JsonValueLike"]
  | typing.Mapping[str, "JsonValueLike"]
  | None
)


@dataclass
class TypedSemanticsStore:
  """Typed Pydantic model store populated alongside raw semantic dictionaries.

  Attributes:
      operations: Validated ``OperationDef`` instances keyed by abstract operation ID.
      framework_traits: Validated ``FrameworkTraits`` instances keyed by framework ID.
  """

  operations: dict[str, OperationDef]
  framework_traits: dict[str, FrameworkTraits]


logger = logging.getLogger(__name__)

# Hierarchy of tier importance for overwriting checks
# Higher numbers mean higher importance (harder to overwrite the Tier Origin)
# EXTRAS is lowest (1) to ensure that if a Neural/Math op is patched in Extras,
# it retains its original high-value semantic tag (e.g. Neural for state injection).
TIER_PRECEDENCE = {
  SemanticTier.NEURAL.value: 3,
  SemanticTier.ARRAY_API.value: 2,
  SemanticTier.NEURAL_OPS.value: 2,
  SemanticTier.EXTRAS.value: 1,
}


def infer_tier_from_priority(priority: int) -> SemanticTier:
  """Map a loading priority integer to a Semantic Tier.

  Args:
      priority (int): Loading order (10=Math, 20=Neural, etc).

  Returns:
      SemanticTier: The inferred tier enum.

  """
  if priority == 10:
    return SemanticTier.ARRAY_API
  if priority == 20:
    return SemanticTier.NEURAL
  return SemanticTier.EXTRAS


def _sync_typed_framework_entry(
  typed_configs: dict[str, FrameworkTraits] | None,
  fw_name: str,
  raw_config: typing.Mapping[str, JsonValue],
) -> None:
  """Validate and store a ``FrameworkTraits`` model into ``typed_configs`` when provided.

  Args:
      typed_configs: Optional typed ``FrameworkTraits`` store to update.
      fw_name: Framework identifier key.
      raw_config: Merged raw configuration dictionary for ``fw_name``.
  """
  if typed_configs is None:
    return
  try:
    typed_configs[fw_name] = FrameworkTraits.model_validate(raw_config)
  except ValidationError as exc:
    logger.debug("Skipping typed FrameworkTraits sync for '%s': %s", fw_name, exc)


def merge_frameworks(
  master_configs: dict[str, dict[str, JsonValue]],
  new_configs: typing.Mapping[str, JsonValue],
  typed_configs: dict[str, FrameworkTraits] | None = None,
) -> None:
  """Merge new framework configurations (from __frameworks__ block) into the master.

  Updates in-place.

  Args:
      master_configs (Dict[str, Dict[str, JsonValue]]): The central framework definitions dictionary.
      new_configs (Mapping[str, JsonValue]): Dictionary of framework traits to merge.
      typed_configs: Optional typed ``FrameworkTraits`` store updated alongside ``master_configs``.

  Raises:
      SemanticLoadError: If a framework configuration entry is not a dictionary.
  """
  for fw_name, traits in new_configs.items():
    if not isinstance(traits, dict):
      raise SemanticLoadError(f"Expected dict for framework '{fw_name}', got {type(traits).__name__}.")
    traits_dict: dict[str, JsonValue] = dict(traits)
    if fw_name not in master_configs:
      master_configs[fw_name] = traits_dict
    else:
      current = master_configs[fw_name]
      if (
        "alias" in traits_dict
        and "alias" in current
        and isinstance(current["alias"], dict)
        and isinstance(traits_dict["alias"], dict)
      ):
        curr_alias = current["alias"]
        new_alias = traits_dict["alias"]
        curr_alias.update(new_alias)
        traits_copy = traits_dict.copy()
        del traits_copy["alias"]
        current.update(traits_copy)
      elif "traits" in traits_dict:
        # If JSON redefined traits, simple merge/overwrite
        current["traits"] = traits_dict["traits"]
      else:
        current.update(traits_dict)
    _sync_typed_framework_entry(typed_configs, fw_name, master_configs[fw_name])


def _try_validate_pattern(raw: JsonValue) -> PatternDef | None:
  """Validate a raw JSON pattern definition into a ``PatternDef`` model.

  Args:
      raw: Raw dictionary or JSON value representing a fusion pattern.

  Returns:
      Validated ``PatternDef`` instance if valid, otherwise ``None``.
  """
  try:
    return PatternDef.model_validate(raw)
  except ValidationError as e:
    logger.warning("⚠️ Invalid pattern definition: %s", e)
    return None


def merge_patterns(
  master_patterns: list[PatternDef],
  new_patterns: typing.Sequence[JsonValue],
) -> None:
  """Append new patterns to the master list, avoiding duplicates by name.

  Args:
      master_patterns (List[PatternDef]): The master list of PatternDef objects to update.
      new_patterns (Sequence[JsonValue]): A list of raw dictionaries representing new patterns.

  Returns:
      None
  """
  existing_names = {p.name for p in master_patterns}

  for raw in new_patterns:
    pat = _try_validate_pattern(raw)
    if pat is not None and pat.name not in existing_names:
      master_patterns.append(pat)
      existing_names.add(pat.name)


def _normalize_args(
  args_list: typing.Sequence[JsonValue],
) -> list[str]:
  """Simplify argument definitions to a list of names for relaxed comparison.

  Converts:
  - ["x", "y"] -> ["x", "y"]
  - [{"name": "x", "type": "int"}, "y"] -> ["x", "y"]

  Args:
      args_list (Sequence[JsonValue]): List of argument definitions (strings, dicts, or tuples).

  Returns:
      List[str]: List of argument names.

  """
  names: list[str] = []
  for arg in args_list:
    if isinstance(arg, str):
      names.append(arg)
    elif isinstance(arg, dict):
      name = arg.get("name")
      if name:
        names.append(str(name))
    elif isinstance(arg, (list, tuple)) and len(arg) > 0:
      names.append(str(arg[0]))
  return names


def _prepare_validated_op(
  op_name: str,
  details: typing.Mapping[str, JsonValueLike] | OperationDef,
  *,
  is_internal: bool,
) -> tuple[OperationDef, dict[str, JsonValue]]:
  """Validate ``details`` into an ``OperationDef`` model and return both model and storage dict.

  Args:
      op_name: Canonical operation identifier.
      details: Raw operation definition mapping or pre-validated ``OperationDef``.
      is_internal: Whether the entry originates from internal defaults.

  Returns:
      Tuple ``(validated_operation_model, serialized_storage_dict)``.
  """
  if isinstance(details, OperationDef):
    validated_op = details.model_copy(update={"is_internal": True}) if is_internal else details
    stored_dict = validated_op.model_dump(by_alias=True, exclude_unset=True)
    stored_dict.setdefault("operation", op_name)
    stored_dict.setdefault("variants", {})
    stored_dict.setdefault("description", validated_op.description or f"Autogenerated definition for {op_name}")
    stored_dict.setdefault("std_args", [])
    if is_internal:
      stored_dict["_is_internal"] = True
    return validated_op, stored_dict

  details_to_validate: dict[str, JsonValueLike] = dict(details)
  details_to_validate.setdefault("operation", op_name)
  details_to_validate.setdefault("variants", {})
  details_to_validate.setdefault("description", f"Autogenerated definition for {op_name}")
  details_to_validate.setdefault("std_args", [])
  if is_internal:
    details_to_validate["_is_internal"] = True

  validated_op = OperationDef.model_validate(details_to_validate)
  validated_dict = validated_op.model_dump(by_alias=True, exclude_unset=True)
  stored_dict = {
    k: v for k, v in details_to_validate.items() if isinstance(v, (int, float, str, bool, list, dict)) or v is None
  }
  stored_dict.update(validated_dict)
  if is_internal:
    stored_dict["_is_internal"] = True
  return validated_op, stored_dict


def _resolve_signature_conflict(
  op_name: str,
  tier: SemanticTier,
  existing_entry: dict[str, JsonValue],
  stored_dict: dict[str, JsonValue],
  *,
  suppress_internal_conflict: bool,
) -> None:
  """Reconcile ``std_args`` differences when two specs at the same tier define ``op_name``.

  Args:
      op_name: Canonical operation identifier.
      tier: Semantic tier being loaded.
      existing_entry: Previously stored operation dictionary.
      stored_dict: Newly validated operation dictionary (mutated on subset downgrade).
      suppress_internal_conflict: Whether internal-to-explicit overwrite warnings should be suppressed.
  """
  raw_old_args = existing_entry.get("std_args", [])
  old_args_raw = raw_old_args if isinstance(raw_old_args, list) else []
  raw_new_args = stored_dict.get("std_args", [])
  new_args_raw = raw_new_args if isinstance(raw_new_args, list) else []

  old_names = _normalize_args(old_args_raw)
  new_names = _normalize_args(new_args_raw)
  if old_names == new_names or suppress_internal_conflict:
    return

  if len(new_names) < len(old_names):
    stored_dict["std_args"] = existing_entry.get("std_args")
    stored_dict["description"] = existing_entry.get("description")
  elif len(new_names) == len(old_names):
    warnings.warn(
      f"Conflict detected for '{op_name}': Signature mismatch within '{tier.value}'. Overwriting.",
      UserWarning,
      stacklevel=2,
    )


def _merge_with_existing_op(
  op_name: str,
  existing_entry: dict[str, JsonValue],
  stored_dict: dict[str, JsonValue],
  prev_tier_val: str,
  tier: SemanticTier,
  *,
  is_internal: bool,
) -> bool:
  """Merge ``stored_dict`` variants/signatures against ``existing_entry`` and decide origin update.

  Args:
      op_name: Canonical operation identifier.
      existing_entry: Previously stored operation dictionary.
      stored_dict: Newly validated operation dictionary (mutated in place).
      prev_tier_val: Previous tier origin string recorded for ``op_name``.
      tier: Semantic tier of the incoming specification.
      is_internal: Whether the incoming specification is marked internal.

  Returns:
      ``True`` if ``key_origins[op_name]`` should be updated to ``tier.value``.
  """
  prev_prec = TIER_PRECEDENCE.get(prev_tier_val, 0)
  curr_prec = TIER_PRECEDENCE.get(tier.value, 0)
  should_update_origin = curr_prec >= prev_prec

  if curr_prec >= prev_prec:
    raw_existing_vars = existing_entry.get("variants")
    existing_variants: dict[str, JsonValue] = dict(raw_existing_vars) if isinstance(raw_existing_vars, dict) else {}
    raw_new_vars = stored_dict.get("variants")
    new_variants: dict[str, JsonValue] = dict(raw_new_vars) if isinstance(raw_new_vars, dict) else {}
    existing_variants.update(new_variants)
    stored_dict["variants"] = existing_variants

    was_internal = bool(existing_entry.get("_is_internal", False))
    if tier != SemanticTier.EXTRAS and should_update_origin and curr_prec == prev_prec:
      _resolve_signature_conflict(
        op_name,
        tier,
        existing_entry,
        stored_dict,
        suppress_internal_conflict=was_internal and not is_internal,
      )

  return should_update_origin


def merge_tier_data(
  data: dict[str, dict[str, JsonValue]],
  key_origins: dict[str, str],
  framework_configs: dict[str, dict[str, JsonValue]],
  new_content: JsonValueLike,
  tier: SemanticTier,
  patterns: list[PatternDef] | None = None,
  is_internal: bool = False,
) -> None:
  """Merge content from a Specification file (hub) into the manager state.

  Handles precedence logic: Neural definitions overwrite Array definitions silently
  for upgrades. Duplicate definitions at the same tier level with conflicting signatures
  trigger warnings only if signatures are ambiguous (same length, different names).
  Otherwise, prefers the richer signature (Superset/Length).

  Args:
      data (Dict[str, Dict[str, JsonValue]]): Master dictionary of operations.
      key_origins (Dict[str, str]): Dict tracking where an op was defined (Math vs Neural).
      framework_configs (Dict[str, Dict[str, JsonValue]]): Master dictionary of framework traits.
      new_content (JsonValue): The JSON content being loaded.
      tier (SemanticTier): The Semantic Tier of the file being loaded.
      patterns (Optional[List[PatternDef]]): Master list of fusion patterns (optional).
      is_internal (bool): If True, marks entries as internal defaults which can be silently overwritten.

  Returns:
      None

  Raises:
      TypeError: If an operation entry in ``new_content`` is not a dictionary.
  """
  if not isinstance(new_content, dict):
    return

  data_copy = dict(new_content)
  if "__frameworks__" in data_copy:
    raw_fw = data_copy.pop("__frameworks__")
    if isinstance(raw_fw, dict):
      merge_frameworks(framework_configs, raw_fw)

  if "__patterns__" in data_copy and patterns is not None:
    raw_pats = data_copy.pop("__patterns__")
    if isinstance(raw_pats, list):
      merge_patterns(patterns, raw_pats)

  for op_name, details in data_copy.items():
    _merge_single_op_entry(
      op_name=op_name,
      details=details,
      data=data,
      key_origins=key_origins,
      tier=tier,
      is_internal=is_internal,
      operations=None,
    )


def _merge_single_op_entry(
  *,
  op_name: str,
  details: JsonValueLike,
  data: dict[str, dict[str, JsonValue]],
  key_origins: dict[str, str],
  tier: SemanticTier,
  is_internal: bool,
  operations: dict[str, OperationDef] | None,
) -> None:
  """Validate and merge one abstract operation entry into ``data`` and optional ``operations``.

  Args:
      op_name: Canonical operation identifier.
      details: Raw mapping or pre-validated ``OperationDef`` model.
      data: Master raw operation dictionary.
      key_origins: Tier-origin map keyed by ``op_name``.
      tier: Semantic tier being merged.
      is_internal: Whether the entry originates from internal defaults.
      operations: Optional typed ``OperationDef`` store updated directly.

  Raises:
      TypeError: If ``details`` is neither a dictionary nor an ``OperationDef``.
  """
  try:
    if not isinstance(details, (dict, OperationDef)):
      raise TypeError(f"Expected mapping for operation '{op_name}', got {type(details).__name__}")
    validated_op, stored_dict = _prepare_validated_op(op_name, details, is_internal=is_internal)

    should_update_origin = True
    if op_name in data:
      if data[op_name] == stored_dict:
        if operations is not None and op_name not in operations:
          operations[op_name] = validated_op
        return
      should_update_origin = _merge_with_existing_op(
        op_name,
        data[op_name],
        stored_dict,
        key_origins.get(op_name, "unknown"),
        tier,
        is_internal=is_internal,
      )
      validated_op = OperationDef.model_validate(stored_dict)

    data[op_name] = stored_dict
    if operations is not None:
      operations[op_name] = validated_op
    if should_update_origin:
      key_origins[op_name] = tier.value
  except ValidationError as e:
    logger.warning("⚠️  Skipping invalid definition '%s' in %s: %s", op_name, tier.value, e)


def merge_tier_with_models(
  typed_store: TypedSemanticsStore,
  data: dict[str, dict[str, JsonValue]],
  key_origins: dict[str, str],
  framework_configs: dict[str, dict[str, JsonValue]],
  new_content: JsonValueLike,
  tier: SemanticTier,
  patterns: list[PatternDef] | None = None,
) -> None:
  """Merge specification content directly into both typed Pydantic stores and raw tables.

  Args:
      typed_store: Typed ``OperationDef`` and ``FrameworkTraits`` container.
      data: Master dictionary of serialized operations.
      key_origins: Dict tracking tier origins per operation.
      framework_configs: Master dictionary of framework configurations.
      new_content: Specification content mapping (supporting ``OperationDef`` values directly).
      tier: Semantic tier of the incoming specification.
      patterns: Optional list of fusion ``PatternDef`` objects.

  Raises:
      TypeError: If an operation entry in ``new_content`` is neither a dict nor ``OperationDef``.
  """
  if not isinstance(new_content, dict):
    return

  data_copy = dict(new_content)
  if "__frameworks__" in data_copy:
    raw_fw = data_copy.pop("__frameworks__")
    if isinstance(raw_fw, dict):
      merge_frameworks(framework_configs, raw_fw, typed_configs=typed_store.framework_traits)

  if "__patterns__" in data_copy and patterns is not None:
    raw_pats = data_copy.pop("__patterns__")
    if isinstance(raw_pats, list):
      merge_patterns(patterns, raw_pats)

  for op_name, details in data_copy.items():
    _merge_single_op_entry(
      op_name=op_name,
      details=details,
      data=data,
      key_origins=key_origins,
      tier=tier,
      is_internal=False,
      operations=typed_store.operations,
    )


def _sync_overlay_operation_model(
  operations: dict[str, OperationDef],
  op_name: str,
  target_framework: str,
  raw_op_dict: typing.Mapping[str, JsonValue],
  implementation: RuntimeTensorOrValue,
) -> None:
  """Update ``operations[op_name]`` with a validated ``FrameworkVariant`` for ``target_framework``.

  Args:
      operations: Typed ``OperationDef`` dictionary to update.
      op_name: Canonical abstract operation identifier.
      target_framework: Framework key whose variant was updated.
      raw_op_dict: Backing operation dictionary (used if ``op_name`` is a new orphan op).
      implementation: Incoming variant implementation (``FrameworkVariant``, ``dict``, or ``None``).
  """
  existing_op = operations.get(op_name)
  if existing_op is None:
    try:
      operations[op_name] = OperationDef.model_validate(raw_op_dict)
    except ValidationError as exc:
      logger.debug("Skipping typed orphan OperationDef sync for '%s': %s", op_name, exc)
    return

  if implementation is None:
    existing_op.variants[target_framework] = None
  elif isinstance(implementation, FrameworkVariant):
    existing_op.variants[target_framework] = implementation
  elif isinstance(implementation, (BaseModel, dict)):
    raw_variants = raw_op_dict.get("variants")
    candidate = raw_variants.get(target_framework) if isinstance(raw_variants, dict) else None
    fallback = implementation.model_dump(exclude_unset=True) if isinstance(implementation, BaseModel) else implementation
    merged_raw = candidate if isinstance(candidate, dict) else fallback
    try:
      existing_op.variants[target_framework] = FrameworkVariant.model_validate(merged_raw)
    except ValidationError as exc:
      logger.debug("Skipping typed FrameworkVariant sync for '%s.%s': %s", op_name, target_framework, exc)


def _resolve_overlay_target_framework(content: typing.Mapping[str, RuntimeTensorOrValue], filename: str) -> str:
  """Determine the target framework key for a snapshot overlay from ``__framework__`` or filename.

  Args:
      content: Snapshot overlay mapping.
      filename: Overlay filename used for ``<fw>_v<ver>`` fallback inference.

  Returns:
      Resolved framework identifier, or empty string if unresolvable.
  """
  raw_target_framework = content.get("__framework__")
  if isinstance(raw_target_framework, str) and raw_target_framework:
    return raw_target_framework
  parts = filename.split("_v")
  return parts[0] if len(parts) > 1 else ""


def _apply_overlay_mapping_entry(
  *,
  op_name: str,
  implementation: RuntimeTensorOrValue,
  target_framework: str,
  filename: str,
  data: dict[str, dict[str, JsonValue]],
  key_origins: dict[str, str],
  typed_store: TypedSemanticsStore | None,
) -> None:
  """Merge a single ``(op_name, implementation)`` overlay entry into ``data`` and ``typed_store``.

  Args:
      op_name: Canonical abstract operation identifier.
      implementation: Variant mapping (``FrameworkVariant``, ``dict``, or ``None``).
      target_framework: Target framework key.
      filename: Source overlay filename for skeleton description generation.
      data: Master raw operation dictionary.
      key_origins: Tier-origin map keyed by ``op_name``.
      typed_store: Optional typed model store updated directly.
  """
  if op_name not in data:
    data[op_name] = {
      "description": f"Auto-generated from {filename}",
      "std_args": [],
      "variants": {},
      "operation": op_name,
    }
    if op_name not in key_origins:
      key_origins[op_name] = (
        SemanticTier.NEURAL.value if (op_name and op_name[0].isupper()) else SemanticTier.EXTRAS.value
      )

  existing_variants = data[op_name].get("variants")
  if not isinstance(existing_variants, dict):
    variants_dict: dict[str, JsonValue] = {}
    data[op_name]["variants"] = variants_dict
  else:
    variants_dict = existing_variants

  impl_dict: dict[str, JsonValue] | None = (
    implementation.model_dump(exclude_unset=True)
    if isinstance(implementation, BaseModel)
    else (dict(implementation) if isinstance(implementation, dict) else None)
  )

  if implementation is None:
    variants_dict[target_framework] = None
  elif impl_dict is not None:
    existing_target = variants_dict.get(target_framework)
    if not isinstance(existing_target, dict):
      current_variant: dict[str, JsonValue] = {}
      variants_dict[target_framework] = current_variant
    else:
      current_variant = existing_target
    current_variant.update(impl_dict)

  if typed_store is not None:
    _sync_overlay_operation_model(typed_store.operations, op_name, target_framework, data[op_name], implementation)


def merge_overlay_data(
  data: dict[str, dict[str, JsonValue]],
  key_origins: dict[str, str],
  framework_configs: dict[str, dict[str, JsonValue]],
  test_templates: dict[str, dict[str, str]],
  content: typing.Mapping[str, RuntimeTensorOrValue],
  filename: str,
  typed_store: TypedSemanticsStore | None = None,
) -> None:
  """Merge a mapping overlay file (snapshot) into the main data.

  Snapshots contain framework-specific implementation overlays ("Spokes")
  that attach to the Abstract Operations ("Hub").

  Args:
      data (Dict[str, Dict[str, JsonValue]]): Master dictionary of operations.
      key_origins (Dict[str, str]): Dict tracking tier origins.
      framework_configs (Dict[str, Dict[str, JsonValue]]): Master dictionary of framework traits.
      test_templates (Dict[str, Dict[str, str]]): Master dictionary of testing templates.
      content (Mapping[str, JsonValue]): The JSON content of the snapshot file.
      filename (str): Filename for metadata inference if needed.
      typed_store: Optional typed ``OperationDef`` and ``FrameworkTraits`` store updated directly.

  Returns:
      None

  Raises:
      SemanticLoadError: If ``content['mappings']`` is not a dictionary.
  """
  target_framework = _resolve_overlay_target_framework(content, filename)
  if not target_framework:
    return

  raw_templates = content.get("templates")
  if isinstance(raw_templates, dict):
    test_templates[target_framework] = {str(k): str(v) for k, v in raw_templates.items()}

  raw_fw_conf = content.get("framework")
  if isinstance(raw_fw_conf, dict):
    if target_framework not in framework_configs:
      framework_configs[target_framework] = dict(raw_fw_conf)
    else:
      framework_configs[target_framework].update(raw_fw_conf)
    if typed_store is not None:
      _sync_typed_framework_entry(typed_store.framework_traits, target_framework, framework_configs[target_framework])

  raw_mappings = content.get("mappings", {})
  if not isinstance(raw_mappings, dict):
    raise SemanticLoadError(f"Overlay '{filename}' mappings must be a dict, got {type(raw_mappings).__name__}.")
  for op_name, implementation in raw_mappings.items():
    _apply_overlay_mapping_entry(
      op_name=op_name,
      implementation=implementation,
      target_framework=target_framework,
      filename=filename,
      data=data,
      key_origins=key_origins,
      typed_store=typed_store,
    )
