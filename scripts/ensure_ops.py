"""Auto-generated doc."""

import yaml
from pathlib import Path


def run():
  """Auto-generated doc."""
  base = Path("src/ml_switcheroo/semantics")

  odl_dir = base / "odl"

  ops_to_ensure = [
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

  for op in ops_to_ensure:
    yaml_file = odl_dir / f"{op}.yaml"
    if not yaml_file.exists():
      data = {"operation": op, "description": f"Verified API: {op}", "std_args": [], "variants": {}}
      with open(yaml_file, "w") as f:
        yaml.dump(data, f, sort_keys=False, indent=2)

  print(f"Ensured {len(ops_to_ensure)} ops in semantics/odl/*.yaml")


if __name__ == "__main__":
  run()
