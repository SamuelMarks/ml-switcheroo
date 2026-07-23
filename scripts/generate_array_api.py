#!/usr/bin/env python3
"""Auto-generated doc."""

import yaml
import sys
import inspect
from pathlib import Path

# Add array-api to path
sys.path.insert(0, str(Path("array-api/src").resolve()))


def run() -> None:
  """Auto-generated doc."""
  odl_dir = Path("src/ml_switcheroo/semantics/odl")
  mapping = {}
  if odl_dir.exists():
    for yml in odl_dir.glob("*.yaml"):
      try:
        with open(yml, "r") as f:
          data = yaml.safe_load(f)
          if data:
            mapping[data.get("operation", yml.stem)] = data
      except Exception:
        pass

  from array_api_stubs import _draft as stub

  ops_set = set()
  for name, obj in inspect.getmembers(stub):
    if not name.startswith("_"):
      ops_set.add(name)

  from array_api_stubs._draft.array_object import array

  for name, obj in inspect.getmembers(array):
    # Include dunder methods
    ops_set.add(name)

  ops = sorted(list(ops_set))
  print(f"Found {len(ops)} Array API ops.")

  for op in ops:
    if op not in mapping:
      mapping[op] = {"description": f"Array API: {op}", "std_args": [], "variants": {}}

  # keep only what's in ops
  new_mapping = {k: v for k, v in mapping.items() if k in ops}

  odl_dir.mkdir(parents=True, exist_ok=True)
  for k, v in new_mapping.items():
    yml = odl_dir / f"{k.replace('/', '_')}.yaml"
    with open(yml, "w") as f:
      yaml.dump(v, f, sort_keys=False, indent=2)

  print(f"Generated {len(new_mapping)} ops in semantics/odl/")


if __name__ == "__main__":
  run()
