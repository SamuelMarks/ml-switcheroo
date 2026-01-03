"""
Core Logic for Framework Definition Injection (JSON).

This module provides the `FrameworkInjector` class, which handles the insertion
or updating of Semantic Operations in the JSON definition files located in
`src/ml_switcheroo/frameworks/definitions/`.

It replaces the legacy LibCST-based injector that modified Python source code.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional

from ml_switcheroo.core.dsl import FrameworkVariant
from ml_switcheroo.frameworks.loader import get_definitions_path
from ml_switcheroo.utils.console import log_info, log_success, log_warning


class FrameworkInjector:
  """
  Injects a `FrameworkVariant` entry into the framework's JSON definitions.

  It handles:
  1. Loading the existing JSON mapping.
  2. Merging the new variant data.
  3. Writing the updated JSON back to disk.
  """

  def __init__(self, target_fw: str, op_name: str, variant: FrameworkVariant):
    """
    Initializes the injector.

    Args:
        target_fw: Key identifier for the framework (e.g., "torch").
        op_name: Abstract Name of the operation key to insert (e.g., "LogSoftmax").
        variant: The ODL Variant definition to serialize.
    """
    self.target_fw = target_fw
    self.op_name = op_name
    self.variant = variant
    self.json_path = get_definitions_path(target_fw)
    self.found = False  # Track if update logic ran successfully

  def inject(self, dry_run: bool = False) -> bool:
    """
    Executes the injection process.

    Args:
        dry_run: If True, prints changes to console instead of writing file.

    Returns:
        bool: True if the operation was successful.
    """
    if not self.json_path.parent.exists():
      if not dry_run:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)

    data = self._load_current()

    # Check idempotency/existing
    new_entry = self.variant.model_dump(exclude_none=True)

    if self.op_name in data:
      if data[self.op_name] == new_entry:
        log_info(f"  {self.target_fw}: Operation '{self.op_name}' already up to date.")
        return True
      log_info(f"  {self.target_fw}: Updating exist definition for '{self.op_name}'.")
    else:
      log_info(f"  {self.target_fw}: Adding new definition for '{self.op_name}'.")

    data[self.op_name] = new_entry
    self.found = True

    if dry_run:
      print(f"[Dry Run] Writing to {self.json_path.name}:")
      print(json.dumps({self.op_name: new_entry}, indent=2))
    else:
      try:
        with open(self.json_path, "w", encoding="utf-8") as f:
          json.dump(data, f, indent=2, sort_keys=True)
        log_success(f"  Updated {self.json_path.name}")
      except OSError as e:
        log_warning(f"Failed to write to {self.json_path}: {e}")
        return False

    return True

  def _load_current(self) -> Dict[str, Any]:
    """
    Safely loads existing JSON data.

    Returns:
        Dict: The current definitions map. Returns empty dict if file missing or corrupt.
    """
    if not self.json_path.exists():
      return {}

    try:
      with open(self.json_path, "r", encoding="utf-8") as f:
        return json.load(f)
    except json.JSONDecodeError:
      log_warning(f"Corrupt JSON at {self.json_path}. Overwriting with new data.")
      return {}
