#!/usr/bin/env python3
"""Validates all semantics/*.json files against SemanticsFile schema."""

import sys
import json
from pathlib import Path

# Add src to sys.path to import ml_switcheroo
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path.resolve()))

try:
  from ml_switcheroo.semantics.schema import SemanticsFile
except ImportError as e:
  print(f"Error importing SemanticsFile: {e}")
  sys.exit(1)


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


def main() -> None:
  """Execute validation for ODL JSON semantics files.

  Expects a list of file paths as command line arguments or defaults
  to scanning src/ml_switcheroo/semantics/*.json.
  Exits with code 1 if any semantics JSON file fails validation.
  """
  files_to_check: list[Path] = []
  if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
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

  if failed:
    sys.exit(1)


if __name__ == "__main__":
  main()
