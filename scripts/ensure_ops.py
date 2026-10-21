"""Ensure baseline ODL operator definitions exist in `src/ml_switcheroo/semantics/odl/`.

Scans the baseline operator list and generates missing ODL YAML definition files
when they are not yet present in the target directory.
"""

import typing
import argparse
import sys
from pathlib import Path

import yaml

BASELINE_OPS: list[str] = [
  "Conv1d",
  "Conv2d",
  "Conv3d",
  "ConvTranspose1d",
  "ConvTranspose2d",
  "ConvTranspose3d",
  "MaxPool1d",
  "MaxPool2d",
  "MaxPool3d",
  "AvgPool1d",
  "AvgPool2d",
  "AvgPool3d",
  "BatchNorm1d",
  "BatchNorm2d",
  "BatchNorm3d",
  "LayerNorm",
  "GroupNorm",
  "Transformer",
  "MultiheadAttention",
  "ReLU",
  "GELU",
  "Swish",
  "SiLU",
  "Linear",
  "Dense",
  "MatMul",
  "Solve",
  "SVD",
  "Add",
  "Sub",
  "Mul",
  "Div",
  "Abs",
  "Exp",
  "Log",
  "Mean",
  "Sum",
  "Max",
  "Min",
]


def run(odl_dir: Path | None = None, ops: typing.Sequence[str] | None = None) -> int:
  """Ensure all baseline operators have corresponding YAML files in `odl_dir`.

  Args:
      odl_dir: Optional path to the `semantics/odl` directory.
      ops: Optional sequence of operator names to ensure.

  Returns:
      Number of newly created operator YAML files.
  """
  target_dir = odl_dir if odl_dir is not None else Path("src/ml_switcheroo/semantics") / "odl"
  ops_to_ensure = list(ops) if ops is not None else BASELINE_OPS
  created = 0

  for op in ops_to_ensure:
    yaml_file = target_dir / f"{op}.yaml"
    if not yaml_file.exists():
      data = {"operation": op, "description": f"Verified API: {op}", "std_args": [], "variants": {}}
      with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, indent=2)
      created += 1

  print(f"Ensured {len(ops_to_ensure)} ops in semantics/odl/*.yaml ({created} created)")
  return created


def main(argv: typing.Sequence[str] | None = None) -> int:
  """Command-line entrypoint for `ensure_ops.py`.

  Args:
      argv: Optional command-line arguments list.

  Returns:
      Exit status code (0 on success).
  """
  parser = argparse.ArgumentParser(description="Ensure baseline ODL operator YAML files exist.")
  parser.add_argument(
    "--odl-dir",
    type=Path,
    default=Path("src/ml_switcheroo/semantics/odl"),
    help="Target ODL YAML directory.",
  )
  parsed = parser.parse_args(argv)
  run(odl_dir=parsed.odl_dir)
  return 0


if __name__ == "__main__":
  sys.exit(main())
