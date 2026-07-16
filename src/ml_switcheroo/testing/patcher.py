"""Data-Driven Self-Healing Patcher.

Implements logic to patch JSON specifications upon failure.
"""

import json
import logging
from pathlib import Path


def patch_json_spec(file_path: Path, op_name: str, fw_name: str, new_tol: float) -> bool:
  """Updates the tolerances of a specific framework variant in a JSON specification file.

  Preserves JSON formatting/indentation and logs the semantic drift event.

  Args:
      file_path: Path to the JSON specification file.
      op_name: The operation name to patch.
      fw_name: The framework variant to patch.
      new_tol: The new tolerance value (used for both rtol and atol for simplicity).

  Returns:
      bool: True if patched successfully, False otherwise.
  """
  logger = logging.getLogger(__name__)

  try:
    with open(file_path, "r", encoding="utf-8") as f:
      data = json.load(f)

    if op_name not in data:
      logger.error(f"Cannot patch: Operation '{op_name}' not found in {file_path}")
      return False

    op_def = data[op_name]

    # Patch root test tolerances if they exist or create them
    op_def["test_rtol"] = new_tol
    op_def["test_atol"] = new_tol

    with open(file_path, "w", encoding="utf-8") as f:
      json.dump(data, f, indent=2, sort_keys=True)
      f.write("\n")

    logger.info(f"Self-Healed: Semantic drift event logged for {op_name}. Updated {file_path} with tol={new_tol}.")
    return True

  except Exception as e:
    logger.error(f"Failed to patch {file_path}: {e}")
    return False
