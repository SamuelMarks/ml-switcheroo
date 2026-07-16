#!/usr/bin/env python3
"""Generate the JSON Schema for the Operation Definition Language (ODL).

This script replaces the old 10k mapping generator with a strict, pure JSON
schema generator that outputs `schema.json` based on the Pydantic models
in `ml_switcheroo.semantics.schema`.
"""

import json
import sys
from pathlib import Path

# Add src to pythonpath
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path.resolve()))

try:
  from ml_switcheroo.semantics.schema import SemanticsFile
except ImportError as e:
  print(f"Error importing SemanticsFile: {e}")
  sys.exit(1)

OUTPUT_FILE = Path(__file__).parent.parent / "src" / "ml_switcheroo" / "semantics" / "schema.json"


def generate_schema() -> None:
  """Generates the JSON schema and writes it to semantics/schema.json."""
  schema = SemanticsFile.model_json_schema()

  with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(schema, f, indent=2, sort_keys=True)
    f.write("\n")

  print(f"Successfully generated JSON schema at {OUTPUT_FILE}")


if __name__ == "__main__":
  generate_schema()
