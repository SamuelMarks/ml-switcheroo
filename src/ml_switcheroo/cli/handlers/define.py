"""Define CLI Command Handler.

This module implements the logic for the `define` CLI command, which takes
a user-provided YAML file, validates it against the Operation Definition Language (ODL)
schema, and installs it into the project's knowledge base.
"""

import shutil
from pathlib import Path
import yaml

from ml_switcheroo.core.dsl import OperationDef
from ml_switcheroo.semantics.paths import resolve_semantics_dir
from ml_switcheroo.utils.console import log_error, log_success


def handle_define(path: Path) -> int:
  """Handles the 'define' command.

  Reads a YAML file, validates it against the ODL schema, and copies it
  to the semantics directory.

  Args:
      path: Path to the ODL YAML file.

  Returns:
      int: 0 on success, 1 on failure.
  """
  if not path.exists():
    log_error(f"File not found: {path}")
    return 1

  try:
    with open(path, "r", encoding="utf-8") as f:
      data = yaml.safe_load(f)

    # Validate against schema
    op_def = OperationDef.model_validate(data)

    # Save to semantics directory
    semantics_dir = resolve_semantics_dir()
    odl_dir = semantics_dir / "odl"
    odl_dir.mkdir(parents=True, exist_ok=True)

    target_path = odl_dir / f"{op_def.operation}.yaml"
    shutil.copy2(path, target_path)

    log_success(f"Successfully defined '{op_def.operation}' and saved to {target_path}")
    return 0

  except Exception as e:
    log_error(f"Failed to define operation: {e}")
    return 1
