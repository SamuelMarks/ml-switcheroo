"""
Merging Logic for Semantic Knowledge Base.

This module handles the aggregation of disparate knowledge sources (Specs, Snapshots)
into the central SemanticsManager. It resolves conflicts based on Tier Precedence
and merging framework-specific configurations.

Handles:
- Merging Specification Tiers (Math, Neural, Extras).
- Merging Snapshot Overlays (Framework mappings).
- Conflict resolution and prioritization (Warning vs Silencing).
- Merging Framework Traits and Import Maps.
- **Merging Patterns**.
"""

import warnings
from typing import Dict, Any, List, Optional, Union

from pydantic import ValidationError

from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.semantics.schema import OperationDef, PatternDef

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
  """
  Maps a loading priority integer to a Semantic Tier.

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


def merge_frameworks(master_configs: Dict[str, Dict], new_configs: Dict[str, Any]) -> None:
  """
  Merges new framework configurations (from __frameworks__ block) into the master.
  Updates in-place.

  Args:
      master_configs: The central framework definitions dictionary.
      new_configs: Dictionary of framework traits to merge.
  """
  for fw_name, traits in new_configs.items():
    if fw_name not in master_configs:
      master_configs[fw_name] = traits
    else:
      current = master_configs[fw_name]
      if "alias" in traits and "alias" in current:
        current["alias"].update(traits["alias"])
        traits_copy = traits.copy()
        del traits_copy["alias"]
        current.update(traits_copy)
      elif "traits" in traits:
        # If JSON redefined traits, simple merge/overwrite
        current["traits"] = traits["traits"]
      else:
        current.update(traits)


def merge_patterns(master_patterns: List[PatternDef], new_patterns: List[Any]) -> None:
  """
  Append new patterns to the master list, avoiding duplicates by name.
  """
  existing_names = {p.name for p in master_patterns}

  for raw in new_patterns:
    try:
      pat = PatternDef.model_validate(raw)
      if pat.name not in existing_names:
        master_patterns.append(pat)
        existing_names.add(pat.name)
    except ValidationError as e:
      print(f"⚠️ Invalid pattern definition: {e}")


def _normalize_args(args_list: List[Any]) -> List[str]:
  """
  Simplifies argument definitions to a list of names for relaxed comparison.

  Converts:
  - ["x", "y"] -> ["x", "y"]
  - [{"name": "x", "type": "int"}, "y"] -> ["x", "y"]

  Args:
      args_list: List of argument definitions (strings, dicts, or tuples).

  Returns:
      List of argument names.
  """
  names = []
  for arg in args_list:
    if isinstance(arg, str):
      names.append(arg)
    elif isinstance(arg, dict):
      name = arg.get("name")
      if name:
        names.append(name)
    elif isinstance(arg, (list, tuple)) and len(arg) > 0:
      names.append(arg[0])
  return names


def merge_tier_data(
  data: Dict[str, Dict],
  key_origins: Dict[str, str],
  framework_configs: Dict[str, Dict],
  new_content: Dict[str, Any],
  tier: SemanticTier,
  patterns: Optional[List[PatternDef]] = None,
  is_internal: bool = False,
) -> None:
  """
  Merges content from a Specification file (hub) into the manager state.

  Handles precedence logic: Neural definitions overwrite Array definitions silently
  for upgrades. Duplicate definitions at the same tier level with conflicting signatures
  trigger warnings only if signatures are ambiguous (same length, different names).
  Otherwise, prefers the richer signature (Superset/Length).

  Args:
      data: Master dictionary of operations.
      key_origins: Dict tracking where an op was defined (Math vs Neural).
      framework_configs: Master dictionary of framework traits.
      new_content: The JSON content being loaded.
      tier: The Semantic Tier of the file being loaded.
      patterns: Master list of fusion patterns (optional).
      is_internal: If True, marks entries as internal defaults which can be silently overwritten.
  """
  # Safety check: Guard against malformed content (e.g. strings or lists)
  if not isinstance(new_content, dict):
    return

  data_copy = new_content.copy()

  if "__frameworks__" in data_copy:
    merge_frameworks(framework_configs, data_copy.pop("__frameworks__"))

  if "__patterns__" in data_copy and patterns is not None:
    merge_patterns(patterns, data_copy.pop("__patterns__"))

  for op_name, details in data_copy.items():
    should_update_origin = True

    try:
      # FIX: Prepare dictionary for validation by injecting fields implied by structure
      details_to_validate = details.copy()

      # 1. Inject missing 'operation' field (redundant with key)
      if "operation" not in details_to_validate:
        details_to_validate["operation"] = op_name

      # 2. Inject missing 'variants' field (Hub specs often lack this)
      if "variants" not in details_to_validate:
        details_to_validate["variants"] = {}

      # 3. Inject missing 'description' field (Required by schema)
      if "description" not in details_to_validate:
        details_to_validate["description"] = f"Autogenerated definition for {op_name}"

      # 4. Inject missing 'std_args' field (Required by schema, default empty)
      if "std_args" not in details_to_validate:
        details_to_validate["std_args"] = []

      validated_op = OperationDef.model_validate(details_to_validate)

      # FIX: We want to store a dictionary that includes valid schema fields
      # BUT PRESERVES extra fields (like 'ver', 'type' in unit tests) that
      # Pydantic strips out if extra='ignore' or implicit.
      validated_dict = validated_op.model_dump(by_alias=True, exclude_unset=True)

      # We base the storage dict on the input details (with injections) to keep extra keys,
      # then overlay the cleaned/validated fields.
      stored_dict = details_to_validate.copy()
      stored_dict.update(validated_dict)

      if is_internal:
        stored_dict["_is_internal"] = True

      if op_name in data:
        # Idempotency Check: If the new definition is identical to existing, skip
        if data[op_name] == stored_dict:
          continue

        existing_entry = data[op_name]
        was_internal = existing_entry.get("_is_internal", False)

        prev_tier_val = key_origins.get(op_name, "unknown")

        # Determine precedence
        prev_prec = TIER_PRECEDENCE.get(prev_tier_val, 0)
        curr_prec = TIER_PRECEDENCE.get(tier.value, 0)

        # Downgrade Protection: Don't let low-priority spec overwrite high-priority origin tag
        # Unless origin was internal default and we are loading file (any precedence)
        if curr_prec < prev_prec:
          should_update_origin = False

        # Intelligent Merging:
        # If we are in the same tier, or upgrading, we should try to preserve existing variants
        # unless the new spec is radically different.
        if curr_prec >= prev_prec:
          existing_variants = existing_entry.get("variants", {})
          new_variants = stored_dict.get("variants", {})

          # Merge variants: New variants overwrite existing ones for the same framework key
          existing_variants.update(new_variants)
          stored_dict["variants"] = existing_variants

          # Warning Logic:
          # Warn only if a meaningful conflict (signature mismatch) exists at the same tier.
          # Upgrades (e.g. Array->Neural) are silent.
          # Overwriting an Internal Default with explicit File content is also silent.
          suppress_internal_conflict = was_internal and not is_internal

          if tier != SemanticTier.EXTRAS and should_update_origin and curr_prec == prev_prec:
            old_args_raw = existing_entry.get("std_args", [])
            new_args_raw = stored_dict.get("std_args", [])

            # Normalize args to names only to check for True mismatch
            # (Ignoring metadata upgrades like type/min/max)
            old_names = _normalize_args(old_args_raw)
            new_names = _normalize_args(new_args_raw)

            if old_names != new_names and not suppress_internal_conflict:
              # Conflict Resolution Strategy: Greatest Argument Count Wins
              # This behaves like 'merging until most complete version'
              if len(new_names) > len(old_names):
                # Upgrade: New spec is richer. Silent overwrite.
                pass
              elif len(new_names) < len(old_names):
                # Downgrade: New spec is a subset. Preserve old spec details.
                stored_dict["std_args"] = existing_entry.get("std_args")
                stored_dict["description"] = existing_entry.get("description")
                # FrameworkVariants were merged above, so we keep the variants from new logic
                pass
              else:
                # Equal length but different names. Ambiguous 1-to-1 conflict.
                warnings.warn(
                  f"Conflict detected for '{op_name}': Signature mismatch within '{tier.value}'. Overwriting.",
                  UserWarning,
                )

      # Merge dictionaries (Overwrite properties with new spec, but variants are merged above)
      data[op_name] = stored_dict

      # Only update the Tier Origin if precedence allows
      if should_update_origin:
        key_origins[op_name] = tier.value

    except ValidationError as e:
      print(f"⚠️  Skipping invalid definition '{op_name}' in {tier.value}: {e}")
      continue


def merge_overlay_data(
  data: Dict[str, Dict],
  key_origins: Dict[str, str],
  framework_configs: Dict[str, Dict],
  test_templates: Dict[str, Dict],
  content: Dict[str, Any],
  filename: str,
) -> None:
  """
  Merges a mapping overlay file (snapshot) into the main data.

  Snapshots contain framework-specific implementation overlays ("Spokes")
  that attach to the Abstract Operations ("Hub").

  Args:
      data: Master dictionary of operations.
      key_origins: Dict tracking tier origins.
      framework_configs: Master dictionary of framework traits.
      test_templates: Master dictionary of testing templates.
      content: The JSON content of the snapshot file.
      filename: Filename for metadata inference if needed.
  """
  target_fw = content.get("__framework__")

  if not target_fw:
    # Fallback: try to guess from filename 'torch_v1.0.json' -> 'torch'
    parts = filename.split("_v")
    if len(parts) > 1:
      target_fw = parts[0]
    else:
      return  # Cannot determine target framework, skip

  # 1. Merge Template Config if present
  if "templates" in content:
    test_templates[target_fw] = content["templates"]

  # 2. Merge Framework Traits (Aliases)
  if "framework" in content:
    if target_fw not in framework_configs:
      framework_configs[target_fw] = content["framework"]
    else:
      framework_configs[target_fw].update(content["framework"])

  # 4. Merge Mappings
  mappings = content.get("mappings", {})
  for op_name, implementation in mappings.items():
    # A. Check if Op exists in Spec
    if op_name not in data:
      # If not in spec, create a skeleton description.
      data[op_name] = {
        "description": f"Auto-generated from {filename}",
        "std_args": [],  # Unknown if not in Spec
        "variants": {},
        "operation": op_name,  # ensure basic struct
      }
      # Heuristic Tier Detection for orphan mappings
      if op_name not in key_origins:
        if op_name and op_name[0].isupper():
          key_origins[op_name] = SemanticTier.NEURAL.value
        else:
          key_origins[op_name] = SemanticTier.EXTRAS.value

    # B. Ensure 'variants' dict exists
    if "variants" not in data[op_name]:
      data[op_name]["variants"] = {}

    # C. Inject FrameworkVariant
    if implementation is None:
      data[op_name]["variants"][target_fw] = None
    else:
      if target_fw not in data[op_name]["variants"]:
        data[op_name]["variants"][target_fw] = {}

      current_variant = data[op_name]["variants"][target_fw]
      # Handle case where current_variant might be None (from previous explicit disable)
      if current_variant is None:
        current_variant = {}
        data[op_name]["variants"][target_fw] = current_variant

      current_variant.update(implementation)
