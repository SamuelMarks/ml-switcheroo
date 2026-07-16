"""JSON Injector for Semantic Specifications.

This module provides the `StandardsInjector`, a utility to update the Semantic
Knowledge Base JSON files (The Hub) with new operation definitions.

It replaces the legacy LibCST-based injector that modified `standards_internal.py`.
"""

import yaml
from typing import Any, Dict, List, Union, Tuple

from ml_switcheroo.core.dsl import OperationDef, ParameterDef
from ml_switcheroo_ir.schema.ghost import SemanticTier
from ml_switcheroo.semantics.paths import resolve_semantics_dir
from ml_switcheroo.utils.console import log_info, log_success


class StandardsInjector:
  """Injects a new operation definition into the Semantic Knowledge Base (JSON).

  It determines the correct JSON file based on naming heuristics or provided tier,
  serializes the `OperationDef` to JSON-compatible dict, and updates the file.
  """

  def __init__(self, op_def: OperationDef, tier: SemanticTier = SemanticTier.EXTRAS):
    """Initializes the injector.

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
    """Executes the injection.

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

    safe_name = op_name.replace("/", "_")
    filename = f"{safe_name}.yaml"
    target_path = resolve_semantics_dir() / "odl" / filename

    data_entry = self._serialize_op(self.op_def)
    data_entry["operation"] = op_name

    if target_path.exists():
      log_info(f"  Updating existing Hub definition for '{op_name}' in {filename}")
    else:
      log_info(f"  Adding new Hub definition for '{op_name}' to {filename}")

    self.found = True

    if dry_run:
      print(f"[Dry Run] Writing to {filename}:\n{yaml.dump(data_entry, indent=2)}")
    else:
      target_path.parent.mkdir(parents=True, exist_ok=True)
      with open(target_path, "w", encoding="utf-8") as f:
        yaml.dump(data_entry, f, sort_keys=False, indent=2, default_flow_style=False)
      log_success(f"  Updated Hub: {filename}")

    return True

  def _serialize_op(self, op: OperationDef) -> Dict[str, Any]:
    """Converts the OperationDef to a JSON-dict optimized for storage."""
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
      out["return_type"] = op.return_type  # type: ignore
    if op.is_inplace:
      out["is_inplace"] = True  # type: ignore
    if op.output_shape_calc:
      out["output_shape_calc"] = op.output_shape_calc

    return out

  def _serialize_args(self, args: List[Union[str, Tuple, Dict, Any]]) -> List[Any]:
    """Normalizes argument list to clean dictionaries or strings."""
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
        result.append(arg)  # type: ignore

    return result
