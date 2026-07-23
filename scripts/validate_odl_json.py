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
  """Auto-generated doc."""
  try:
    with open(filepath, "r", encoding="utf-8") as f:
      content = json.load(f)
    SemanticsFile.model_validate(content)
    return True
  except Exception as e:
    print(f"Validation failed for {filepath}: {e}")
    return False


def main() -> None:
  """Auto-generated doc."""
  failed = False
  for arg in sys.argv[1:]:
    filepath = Path(arg)
    if filepath.suffix == ".json" and "semantics" in filepath.parts:
      if not validate_file(filepath):
        failed = True

  if failed:
    sys.exit(1)


if __name__ == "__main__":
  main()
