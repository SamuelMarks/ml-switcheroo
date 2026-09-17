#!/usr/bin/env python3
"""Validates all semantics/*.json files against SemanticsFile schema."""

import json
from pathlib import Path
import sys
from typing import List, Optional, Sequence

# Add src to sys.path to import ml_switcheroo
src_path = Path(__file__).parent.parent / "src"
if str(src_path.resolve()) not in sys.path:
  sys.path.insert(0, str(src_path.resolve()))

from ml_switcheroo.semantics.schema import SemanticsFile  # noqa: E402


def validate_file(filepath: Path) -> bool:
  """Validate a single JSON file against the SemanticsFile schema.

  Args:
      filepath: Path to the JSON file to validate.

  Returns:
      True if the file is valid, False otherwise.
  """
  try:
    with open(filepath, "r", encoding="utf-8") as f:
      content = json.load(f)
    SemanticsFile.model_validate(content)
    return True
  except Exception as e:
    print(f"Validation failed for {filepath}: {e}")
    return False


def main(argv: Optional[Sequence[str]] = None) -> int:
  """Execute validation for ODL JSON semantics files.

  Expects a list of file paths as command line arguments or defaults
  to scanning src/ml_switcheroo/semantics/*.json.
  Exits with code 1 if any semantics JSON file fails validation.

  Args:
      argv: Optional command-line arguments list.

  Returns:
      Exit code (0 for success, 1 for failure).
  """
  cli_args = list(sys.argv[1:] if argv is None else argv)
  files_to_check: List[Path] = []

  if len(cli_args) > 0:
    for arg in cli_args:
      filepath = Path(arg)
      if filepath.suffix == ".json" and "semantics" in filepath.parts:
        files_to_check.append(filepath)
  else:
    sem_dir = src_path / "ml_switcheroo" / "semantics"
    for jf in sem_dir.glob("*.json"):
      if jf.name not in ("priority_scores.json", "nvidia_sass_isa.json", "rdna_isa.json"):
        files_to_check.append(jf)

  failed = False
  for filepath in files_to_check:
    if not validate_file(filepath):
      failed = True

  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
