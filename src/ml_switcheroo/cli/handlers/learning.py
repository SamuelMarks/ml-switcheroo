"""
Interactive and Learning Command Handlers.

This module provides tools for:
1.  **Wizard**: Interactive CLI to categorize unmapped APIs.
2.  **Harvest**: Learning mappings from manual test files.
"""

from pathlib import Path

from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.cli.wizard import MappingWizard
from ml_switcheroo.discovery.harvester import SemanticHarvester
from ml_switcheroo.utils.console import (
  log_info,
  log_success,
  log_error,
)


def handle_wizard(package: str) -> int:
  """
  Handles the 'wizard' command for interactive mapping discovery.

  Args:
      package: The name of the python package to inspect (e.g., 'torch').

  Returns:
      int: Exit code (0 for success).
  """
  semantics = SemanticsManager()
  wizard = MappingWizard(semantics)
  wizard.start(package)
  return 0


def handle_harvest(path: Path, target: str, dry_run: bool) -> int:
  """
  Handles the 'harvest' command to learn mappings from manual tests.

  Args:
      path: File or directory containing python test files.
      target: The framework target used in the tests (e.g., 'jax').
      dry_run: If True, prints changes without writing to disk.

  Returns:
      int: Exit code (0 for success, 1 for path errors).
  """
  semantics = SemanticsManager()
  harvester = SemanticHarvester(semantics, target_fw=target)
  files = []
  if path.is_file():
    files.append(path)
  elif path.is_dir():
    files.extend(list(path.rglob("test_*.py")))
  else:
    log_error(f"Invalid path: {path}")
    return 1

  total_updated = 0
  for f in files:
    total_updated += harvester.harvest_file(f, dry_run=dry_run)

  if total_updated > 0 and not dry_run:
    log_success(f"Harvest complete. Updated {total_updated} definitions.")
  elif total_updated == 0:
    log_info("No new manual fixes found to harvest.")
  return 0
