"""
Merging Logic for Semantic Knowledge Base.

Handles:
- Merging Specification Tiers (Math, Neural, Extras).
- Merging Snapshot Overlays (Framework mappings).
- Conflict resolution and prioritization.
"""

import warnings
from typing import Dict, Any

from pydantic import ValidationError

from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.semantics.schema import OpDefinition


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


def merge_imports(master_import_data: Dict[str, Dict], new_imports: Dict[str, Any]) -> None:
  """
  Merges new import definitions (from __imports__ block) into the master dictionary.
  Updates in-place.
  """
  for src_mod, details in new_imports.items():
    if src_mod not in master_import_data:
      master_import_data[src_mod] = details
    else:
      existing_variants = master_import_data[src_mod].get("variants", {})
      new_variants = details.get("variants", {})
      existing_variants.update(new_variants)
      master_import_data[src_mod]["variants"] = existing_variants


def merge_frameworks(master_configs: Dict[str, Dict], new_configs: Dict[str, Any]) -> None:
  """
  Merges new framework configurations (from __frameworks__ block) into the master.
  Updates in-place.
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


def merge_tier_data(
  data: Dict[str, Dict],
  key_origins: Dict[str, str],
  import_data: Dict[str, Dict],
  framework_configs: Dict[str, Dict],
  new_content: Dict[str, Any],
  tier: SemanticTier,
) -> None:
  """
  Merges content from a Specification file (hub) into the manager state.

  Args:
      data: Master dictionary of operations.
      key_origins: Dict tracking where an op was defined (Math vs Neural).
      import_data: Master dictionary of import mappings.
      framework_configs: Master dictionary of framework traits.
      new_content: The JSON content being loaded.
      tier: The Semantic Tier of the file being loaded.
  """
  data_copy = new_content.copy()

  if "__imports__" in data_copy:
    merge_imports(import_data, data_copy.pop("__imports__"))

  if "__frameworks__" in data_copy:
    merge_frameworks(framework_configs, data_copy.pop("__frameworks__"))

  for op_name, details in data_copy.items():
    if op_name in data:
      if tier != SemanticTier.EXTRAS:
        prev_tier = key_origins.get(op_name, "unknown")
        warnings.warn(
          f"Conflict detected for '{op_name}': Defined in '{prev_tier}' but overwritten by '{tier.value}' in load.",
          UserWarning,
        )

    try:
      validated_op = OpDefinition.model_validate(details)
      stored_dict = validated_op.model_dump(by_alias=True, exclude_unset=True)
      data[op_name] = stored_dict
      key_origins[op_name] = tier.value
    except ValidationError as e:
      print(f"⚠️  Skipping invalid definition '{op_name}' in {tier.value}: {e}")
      continue


def merge_overlay_data(
  data: Dict[str, Dict],
  key_origins: Dict[str, str],
  import_data: Dict[str, Dict],
  framework_configs: Dict[str, Dict],
  test_templates: Dict[str, Dict],
  content: Dict[str, Any],
  filename: str,
) -> None:
  """
  Merges a mapping overlay file (snapshot) into the main data.

  Args:
      data: Master dictionary of operations.
      key_origins: Dict tracking tier origins.
      import_data: Master dictionary of import mappings.
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

  # 3. Merge Import Maps
  if "imports" in content:
    # Handle inversion: Snapshot has { "torch.nn": { "root": "flax", ... } }
    # Manager structure is { "torch.nn": { "variants": { "jax": { ... } } } }
    for src_mod, details in content["imports"].items():
      if src_mod not in import_data:
        import_data[src_mod] = {"variants": {}}

      import_data[src_mod]["variants"][target_fw] = details

  # 4. Merge Mappings
  mappings = content.get("mappings", {})
  for op_name, implementation in mappings.items():
    # A. Check if Op exists in Spec
    if op_name not in data:
      # If not in spec, create a skeleton Extra.
      data[op_name] = {
        "description": f"Auto-generated from {filename}",
        "std_args": [],  # Unknown if not in Spec
        "variants": {},
      }
      key_origins[op_name] = SemanticTier.EXTRAS.value

    # B. Ensure 'variants' dict exists
    if "variants" not in data[op_name]:
      data[op_name]["variants"] = {}

    # C. Inject Variant
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
