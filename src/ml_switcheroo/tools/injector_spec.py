"""
JSON Injector for Semantic Specifications.

This module provides the `StandardsInjector`, a utility to update the Semantic
Knowledge Base JSON files (The Hub) with new operation definitions.

It replaces the legacy LibCST-based injector that modified `standards_internal.py`.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple

from ml_switcheroo.core.dsl import OperationDef, ParameterDef
from ml_switcheroo.enums import SemanticTier
from ml_switcheroo.semantics.paths import resolve_semantics_dir
from ml_switcheroo.utils.console import log_info, log_success, log_warning


class StandardsInjector:
  """
  Injects a new operation definition into the Semantic Knowledge Base (JSON).

  It determines the correct JSON file based on naming heuristics or provided tier,
  serializes the `OperationDef` to JSON-compatible dict, and updates the file.
  """

  def __init__(self, op_def: OperationDef, tier: SemanticTier = SemanticTier.EXTRAS):
    """
    Initializes the injector.

    Args:
        op_def: The definition model containing metadata and signatures.
        tier: The target semantic tier (default: EXTRAS).
              Heuristics in `inject()` may override this if the name suggests
              a Neural operation.
    """
    self.op_def = op_def
    self.tier = tier
    self.found = False

  def inject(self, dry_run: bool = False) -> bool:
    """
    Executes the injection.

    Args:
        dry_run: If True, prints intended changes without writing to disk.

    Returns:
        bool: True on success.
    """
    # 1. Determine Tier / Filename
    # Heuristic: Start with uppercase (PascalCase) usually implies Neural/Class
    op_name = self.op_def.operation

    if op_name[0].isupper() and self.tier == SemanticTier.EXTRAS:
      # Simple heuristic: "Conv2d" -> Neural, "abs" -> Math
      self.tier = SemanticTier.NEURAL

    # NOTE: Removed islower() heuristic that forced EXTRAS->ARRAY_API.
    # Explicit EXTRAS assignment should be respected for utilities like 'save' or 'load'.

    if self.tier == SemanticTier.ARRAY_API:
      filename = "k_array_api.json"
    elif self.tier == SemanticTier.NEURAL:
      filename = "k_neural_net.json"
    else:
      filename = "k_framework_extras.json"

    target_path = resolve_semantics_dir() / filename

    # 2. Serialize Definition
    # We manually construct the dict to ensure clean output matching the schema
    # `model_dump` often includes null fields we want to omit for brevity
    data_entry = self._serialize_op(self.op_def)

    # 3. Load Existing
    current_data = {}
    if target_path.exists():
      try:
        with open(target_path, "r", encoding="utf-8") as f:
          current_data = json.load(f)
      except json.JSONDecodeError:
        log_warning(f"Corrupt JSON at {target_path}. Proceeding with empty dict.")

    # 4. Update
    if op_name in current_data:
      log_info(f"  Updating existing Hub definition for '{op_name}' in {filename}")
    else:
      log_info(f"  Adding new Hub definition for '{op_name}' to {filename}")

    current_data[op_name] = data_entry
    self.found = True

    # 5. Write
    if dry_run:
      print(f"[Dry Run] Writing to {filename}:\n{json.dumps({op_name: data_entry}, indent=2)}")
    else:
      if not target_path.parent.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
      with open(target_path, "w", encoding="utf-8") as f:
        json.dump(current_data, f, indent=2, sort_keys=True)
      log_success(f"  Updated Hub: {filename}")

    return True

  def _serialize_op(self, op: OperationDef) -> Dict[str, Any]:
    """
    Converts the OperationDef to a JSON-dict optimized for storage.
    """
    # Basic fields
    out = {
      "description": op.description,
      "std_args": self._serialize_args(op.std_args),
      "variants": {},  # Hub only stores abstract spec, mapping is in Spoke/Snapshot
    }

    # Optional fields (only add if not default)
    if op.op_type != "function":
      out["op_type"] = op.op_type
    if op.return_type != "Any":
      out["return_type"] = op.return_type
    if op.is_inplace:
      out["is_inplace"] = True
    if op.output_shape_calc:
      out["output_shape_calc"] = op.output_shape_calc

    return out

  def _serialize_args(self, args: List[Union[str, Tuple, Dict, Any]]) -> List[Any]:
    """
    Normalizes argument list to clean dictionaries or strings.
    """
    result = []
    for arg in args:
      if isinstance(arg, (ParameterDef, dict)):
        # Convert object/dict to clean dict
        if isinstance(arg, ParameterDef):
          d = arg.model_dump(exclude_none=True)
        else:
          d = arg.copy()
          # Filter None values manually if it was a raw dict
          d = {k: v for k, v in d.items() if v is not None}

        # Simplify: if it only has name and type='Any', store as string?
        # No, stick to dicts for consistency if provided as such.
        result.append(d)

      elif isinstance(arg, (list, tuple)):
        # Legacy tuple ["x", "type"]
        entry = {"name": arg[0]}
        if len(arg) > 1:
          entry["type"] = arg[1]
        result.append(entry)

      elif isinstance(arg, str):
        result.append(arg)

    return result
