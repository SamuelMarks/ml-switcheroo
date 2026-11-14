"""Audits and normalizes ODL YAML definitions and quarantine JSON artifacts.

Separates verified operator specifications from unverified entries and ensures
consistent ``std_args`` and ``variants`` structure across ``semantics/odl/*.yaml``.
"""

import json
import logging
from pathlib import Path

import yaml
from pydantic import JsonValue

_CORE_VERIFIED_APIS: frozenset[str] = frozenset(
  {
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
)


def _load_single_odl_entry(yaml_file: Path) -> tuple[str, dict[str, JsonValue]] | None:
  """Load and parse a single discrete ODL YAML file.

  Args:
      yaml_file: Path to the discrete ODL ``*.yaml`` file.

  Returns:
      Tuple of ``(operation_name, spec_dict)`` if readable and valid, otherwise ``None``.
  """
  try:
    with open(yaml_file, "r") as yaml_f:
      data: JsonValue = yaml.safe_load(yaml_f)
      if isinstance(data, dict):
        raw_op = data.get("operation", yaml_file.stem)
        op_name: str = str(raw_op) if isinstance(raw_op, str) else yaml_file.stem
        return op_name, data
  except (OSError, yaml.YAMLError, ValueError, TypeError) as exc:
    logging.getLogger(__name__).debug("Skipping unreadable ODL file %s: %s", yaml_file, exc)
  return None


def _load_odl_entries(odl_dir: Path) -> dict[str, dict[str, JsonValue]]:
  """Load discrete ODL YAML files from ``odl_dir`` into an operation dictionary.

  Args:
      odl_dir: Directory containing discrete ODL ``*.yaml`` files.

  Returns:
      Mapping of operation identifier to parsed YAML dictionary.
  """
  entries: dict[str, dict[str, JsonValue]] = {}
  if not odl_dir.exists():
    return entries
  for yaml_file in odl_dir.glob("*.yaml"):
    loaded = _load_single_odl_entry(yaml_file)
    if loaded is not None:
      entries[loaded[0]] = loaded[1]
  return entries


def _normalize_op_schema(op_spec: dict[str, JsonValue]) -> None:
  """Normalize ``std_args`` and ``variants`` dictionaries in-place on ``op_spec``.

  Args:
      op_spec: Operation specification dictionary to standardize.
  """
  new_args: list[JsonValue] = []
  raw_std_args = op_spec.get("std_args", [])
  if isinstance(raw_std_args, list):
    for arg in raw_std_args:
      if isinstance(arg, str):
        arg = {"name": arg, "type": "Any"}
      elif isinstance(arg, dict):
        arg.setdefault("name", "unknown")
        arg.setdefault("type", "Any")
      new_args.append(arg)
  op_spec["std_args"] = new_args

  raw_variants = op_spec.get("variants", {})
  if isinstance(raw_variants, dict):
    for var in raw_variants.values():
      if isinstance(var, dict):
        var.setdefault("api", "")
        var.setdefault("args", {})


def _write_odl_and_quarantine(
  odl_dir: Path,
  quarantine_file: Path,
  verified_ops: dict[str, dict[str, JsonValue]],
  quarantine_ops: dict[str, dict[str, JsonValue]],
) -> None:
  """Write normalized verified ODL YAML files and ``quarantine.json``.

  Args:
      odl_dir: Target ODL YAML directory.
      quarantine_file: Target ``quarantine.json`` path.
      verified_ops: Mapping of verified operation definitions.
      quarantine_ops: Mapping of quarantined operation definitions.
  """
  if odl_dir.exists():
    for existing_file in odl_dir.glob("*.yaml"):
      existing_file.unlink()

  for op_key, op_val in verified_ops.items():
    op_val["operation"] = op_key
    out_yaml_file: Path = odl_dir / f"{op_key.replace('/', '_')}.yaml"
    with open(out_yaml_file, "w") as out_f:
      yaml.dump(op_val, out_f, sort_keys=False, indent=2)

  with open(quarantine_file, "w") as qu_f:
    json.dump(quarantine_ops, qu_f, indent=2)


def run(extra_ops: dict[str, dict[str, JsonValue]] | None = None) -> None:
  """Audit ODL YAML specifications, normalize argument schemas, and update ``quarantine.json``.

  Args:
      extra_ops: Optional mapping of extra operations to include in quarantine.
  """
  sem_dir: Path = Path("src/ml_switcheroo/semantics")
  odl_dir: Path = sem_dir / "odl"
  quarantine_file: Path = sem_dir / "quarantine.json"

  k_nn = _load_odl_entries(odl_dir)
  quarantine: dict[str, dict[str, JsonValue]] = {}
  new_k_nn: dict[str, dict[str, JsonValue]] = {}

  verified_apis = set(_CORE_VERIFIED_APIS) | {k for k, v in k_nn.items() if v.get("variants")}
  for k, v in k_nn.items():
    if k in verified_apis:
      _normalize_op_schema(v)
      new_k_nn[k] = v
    else:
      quarantine[k] = v

  if extra_ops is not None:
    quarantine.update(extra_ops)

  _write_odl_and_quarantine(odl_dir, quarantine_file, new_k_nn, quarantine)


if __name__ == "__main__":
  run()
