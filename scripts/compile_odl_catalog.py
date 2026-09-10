"""ODL Catalog Compiler and Roundtrip Verifier.

Compiles discrete ODL YAML files into a unified JSON catalog (`odl.json`),
validates against the declarative SemanticsFile schema, and verifies
bidirectional serialization roundtrips.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
  sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from ml_switcheroo.semantics.schema import SemanticsFile  # noqa: E402


def compile_catalog(
  odl_dir: Path,
  output_json: Path,
  validate: bool = True,
) -> int:
  """Compiles all discrete ODL YAML files into a single JSON catalog.

  Args:
      odl_dir: Path to directory containing ODL YAML files.
      output_json: Path to target compiled JSON file.
      validate: If True, validates compiled catalog against SemanticsFile schema.

  Returns:
      Number of operations compiled.

  Raises:
      ValueError: If validation against SemanticsFile fails.
  """
  catalog: Dict[str, Any] = {}

  for yaml_path in sorted(odl_dir.glob("*.yaml")):
    try:
      with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    except Exception:
      continue

    if not isinstance(data, dict):
      continue

    op_name = data.get("operation", yaml_path.stem)
    catalog[op_name] = data

  if validate:
    try:
      SemanticsFile.model_validate(catalog)
    except Exception as e:
      raise ValueError(f"Compiled catalog failed SemanticsFile validation: {e}") from e

  output_json.parent.mkdir(parents=True, exist_ok=True)
  with open(output_json, "w", encoding="utf-8") as f:
    json.dump(catalog, f, indent=2, sort_keys=True)
    f.write("\n")

  return len(catalog)


def verify_roundtrip(odl_dir: Path, json_path: Path) -> bool:
  """Verifies serialization roundtrip between discrete YAMLs and compiled JSON.

  Args:
      odl_dir: Path to directory containing ODL YAML files.
      json_path: Path to compiled JSON catalog file.

  Returns:
      True if roundtrip equivalence is verified, False otherwise.
  """
  if not json_path.is_file():
    return False

  with open(json_path, "r", encoding="utf-8") as f:
    compiled_data = json.load(f)

  if not isinstance(compiled_data, dict):
    return False

  yaml_ops: Dict[str, Any] = {}
  for yaml_path in odl_dir.glob("*.yaml"):
    try:
      with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
      if isinstance(data, dict):
        op_name = data.get("operation", yaml_path.stem)
        yaml_ops[op_name] = data
    except Exception:
      continue

  if set(yaml_ops.keys()) != set(compiled_data.keys()):
    return False

  for k, yaml_val in yaml_ops.items():
    if compiled_data.get(k) != yaml_val:
      return False

  return True


def decompile_catalog(json_path: Path, output_dir: Path) -> int:
  """Decompiles a compiled JSON catalog back into discrete YAML files.

  Args:
      json_path: Path to the compiled JSON catalog file.
      output_dir: Directory where discrete YAML files will be written.

  Returns:
      Number of discrete YAML files created.
  """
  with open(json_path, "r", encoding="utf-8") as f:
    catalog = json.load(f)

  if not isinstance(catalog, dict):
    return 0

  output_dir.mkdir(parents=True, exist_ok=True)
  count = 0
  for op_name, op_spec in catalog.items():
    if not isinstance(op_spec, dict):
      continue
    target_yaml = output_dir / f"{op_name}.yaml"
    with open(target_yaml, "w", encoding="utf-8") as f:
      yaml.dump(op_spec, f, sort_keys=False, indent=2)
    count += 1

  return count


def main(args: Optional[Sequence[str]] = None) -> int:
  """Command-line interface for ODL catalog compilation and roundtrip verification.

  Args:
      args: Optional command-line argument list.

  Returns:
      Exit code (0 on success).
  """
  parser = argparse.ArgumentParser(description="Compile and verify ODL catalogs")
  parser.add_argument(
    "--odl-dir",
    type=Path,
    default=Path("src/ml_switcheroo/semantics/odl"),
    help="Path to ODL YAML files",
  )
  parser.add_argument(
    "--output-json",
    type=Path,
    default=Path("src/ml_switcheroo/semantics/odl.json"),
    help="Path to target compiled JSON catalog",
  )
  parser.add_argument(
    "--verify-only",
    action="store_true",
    help="Verify roundtrip equivalence without compiling",
  )
  parsed = parser.parse_args(args)

  if parsed.verify_only:
    ok = verify_roundtrip(parsed.odl_dir, parsed.output_json)
    if ok:
      print("✅ Roundtrip verification passed.")
      return 0
    print("❌ Roundtrip verification failed.")
    return 1

  count = compile_catalog(parsed.odl_dir, parsed.output_json, validate=True)
  print(f"Compiled and validated {count} operations into {parsed.output_json}.")

  ok = verify_roundtrip(parsed.odl_dir, parsed.output_json)
  if not ok:
    print("❌ Roundtrip verification failed after compilation.")
    return 1

  print("✅ Roundtrip serialization verified.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
