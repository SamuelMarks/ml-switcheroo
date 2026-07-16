"""Auto-generated doc."""

import json
import yaml
from pathlib import Path


def run():
  """Auto-generated doc."""
  sem_dir = Path("src/ml_switcheroo/semantics")
  odl_dir = sem_dir / "odl"
  quarantine_file = sem_dir / "quarantine.json"

  k_nn = {}
  k_extras = {}

  if odl_dir.exists():
    for yaml_file in odl_dir.glob("*.yaml"):
      try:
        with open(yaml_file, "r") as f:
          data = yaml.safe_load(f)
          if data:
            op_name = data.get("operation", yaml_file.stem)
            k_nn[op_name] = data
      except Exception:
        pass

  verified_apis = {
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
  }

  # Also keep APIs that have non-empty variants
  for k, v in k_nn.items():
    if v.get("variants"):
      verified_apis.add(k)

  quarantine = {}
  new_k_nn = {}

  for k, v in k_nn.items():
    if k in verified_apis:
      new_k_nn[k] = v
    else:
      quarantine[k] = v

  for k, v in k_extras.items():
    quarantine[k] = v

  # Standardize args and variants
  for k, v in new_k_nn.items():
    new_args = []
    for arg in v.get("std_args", []):
      if isinstance(arg, str):
        arg = {"name": arg, "type": "Any"}
      elif isinstance(arg, dict):
        if "name" not in arg:
          arg["name"] = "unknown"
        if "type" not in arg:
          arg["type"] = "Any"
      new_args.append(arg)
    v["std_args"] = new_args

    for fw, var in v.get("variants", {}).items():
      if "api" not in var:
        var["api"] = ""
      if "args" not in var:
        var["args"] = {}

  # Rewrite odl files
  if odl_dir.exists():
    for f in odl_dir.glob("*.yaml"):
      f.unlink()

  for k, v in new_k_nn.items():
    yaml_file = odl_dir / f"{k.replace('/', '_')}.yaml"
    with open(yaml_file, "w") as f:
      yaml.dump(v, f, sort_keys=False, indent=2)

  with open(quarantine_file, "w") as f:
    json.dump(quarantine, f, indent=2)


if __name__ == "__main__":
  run()
