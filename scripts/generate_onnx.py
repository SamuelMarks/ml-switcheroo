#!/usr/bin/env python3
"""Auto-generated doc."""

import json
from pathlib import Path
import onnx


def run() -> None:
  """Auto-generated doc."""
  out_file = Path("src/ml_switcheroo/frameworks/definitions/onnx.json")
  if out_file.exists():
    with open(out_file, "r") as f:
      mapping = json.load(f)
  else:
    mapping = {}

  schemas = onnx.defs.get_all_schemas()  # type: ignore[attr-defined]
  print(f"Found {len(schemas)} ONNX schemas.")

  for s in schemas:
    op_name = s.name
    # Use existing if mapped, otherwise add empty
    if op_name not in mapping:
      # Map standard inputs to std_args roughly
      args_mapping = {}
      for idx, inp in enumerate(s.inputs):
        args_mapping[inp.name] = inp.name

      for attr in s.attributes.values():
        args_mapping[attr.name] = attr.name

      mapping[op_name] = {
        "api": f"onnx.{op_name}",
        "args": args_mapping,
        "pack_as": "Tuple",
        "dispatch_rules": [],
        "required_imports": [],
      }

  with open(out_file, "w") as f:
    json.dump(mapping, f, indent=2, sort_keys=True)
  print(f"Generated {len(mapping)} ops in {out_file}")


if __name__ == "__main__":
  run()
