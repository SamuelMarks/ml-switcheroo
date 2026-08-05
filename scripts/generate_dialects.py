"""Script to generate dialects."""

import json


def generate() -> None:
  """Generate dialects."""
  with open("mlir_official_ops.json") as f:
    data = json.load(f)

  out = [
    '"""Generated MLIR Official Dialects.',
    "",
    "This module contains automatically generated OpSchemas for all official MLIR operations.",
    '"""',
    "",
    "from typing import Dict",
    "from ml_switcheroo.core.mlir.dialect import OpSchema",
    "",
    "OFFICIAL_OPS: Dict[str, OpSchema] = {",
  ]

  for dialect, ops in data.items():
    for op in ops:
      name = f"{dialect}.{op}"
      out.append(f'  "{name}": OpSchema(name="{name}"),')

  out.append("}")
  out.append("")

  with open("src/ml_switcheroo/core/mlir/official_dialects.py", "w") as f:
    f.write("\n".join(out))


if __name__ == "__main__":
  generate()
