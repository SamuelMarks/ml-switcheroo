#!/usr/bin/env python3
"""Validates all semantics/*.json files against SemanticsFile schema."""

import json
import sys
import typing
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str((Path(__file__).parent.parent / "src").resolve()))

from ml_switcheroo.semantics.schema import SemanticsFile

src_path = Path(__file__).parent.parent / "src"


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
  except (OSError, json.JSONDecodeError, ValidationError, ValueError, TypeError) as e:
    print(f"Validation failed for {filepath}: {e}")
    return False


def main(argv: typing.Sequence[str] | None = None) -> int:
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

  if len(cli_args) > 0:
    candidates = [Path(arg) for arg in cli_args]
    files_to_check = [fp for fp in candidates if fp.suffix == ".json" and "semantics" in fp.parts]
  else:
    sem_dir = src_path / "ml_switcheroo" / "semantics"
    files_to_check = [
      jf
      for jf in sem_dir.glob("*.json")
      if jf.name not in ("priority_scores.json", "nvidia_sass_isa.json", "rdna_isa.json")
    ]

  failed = False
  for filepath in files_to_check:
    if not validate_file(filepath):
      failed = True

  return 1 if failed else 0


if __name__ == "__main__":
  sys.exit(main())
