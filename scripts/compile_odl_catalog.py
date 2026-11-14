"""ODL Catalog Compiler and Roundtrip Verifier.

Compiles discrete ODL YAML files into a unified JSON catalog (``odl.json``) and
optional deterministic documentation YAML catalog, validates against the
declarative ``SemanticsFile`` schema, and verifies bidirectional serialization
roundtrips with zero stale duplicate catalogs.
"""

import argparse
import json
import logging
import sys
import typing
from pathlib import Path

import yaml
from pydantic import JsonValue, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml_switcheroo.semantics.schema import SemanticsFile

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_odl_directory(odl_dir: Path) -> dict[str, dict[str, JsonValue]]:
  """Load all valid discrete ODL YAML files in deterministic key order.

  Args:
      odl_dir: Path to directory containing discrete ODL ``*.yaml`` files.

  Returns:
      Mapping of operation name to parsed operation specification dictionary.
  """
  catalog: dict[str, dict[str, JsonValue]] = {}
  for yaml_path in sorted(odl_dir.glob("*.yaml")):
    try:
      with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError, ValueError, RuntimeError) as exc:
      logging.getLogger(__name__).debug("Skipping unreadable ODL YAML %s: %s", yaml_path, exc)
      continue

    if not isinstance(data, dict):
      continue

    op_name = str(data.get("operation", yaml_path.stem))
    catalog[op_name] = data
  return catalog


def compile_catalog(
  odl_dir: Path,
  output_json: Path,
  validate: bool = True,
) -> int:
  """Compile all discrete ODL YAML files into a single JSON catalog.

  Args:
      odl_dir: Path to directory containing ODL YAML files.
      output_json: Path to target compiled JSON file.
      validate: If True, validates compiled catalog against ``SemanticsFile`` schema.

  Returns:
      Number of operations compiled.

  Raises:
      ValueError: If validation against ``SemanticsFile`` fails.
  """
  catalog = _load_odl_directory(odl_dir)

  if validate:
    try:
      SemanticsFile.model_validate(catalog)
    except (ValidationError, ValueError, TypeError) as exc:
      raise ValueError(f"Compiled catalog failed SemanticsFile validation: {exc}") from exc

  output_json.parent.mkdir(parents=True, exist_ok=True)
  with open(output_json, "w", encoding="utf-8") as f:
    print(json.dumps(catalog, indent=2, sort_keys=True), file=f)

  return len(catalog)


def export_docs_catalog(odl_dir: Path, output_yaml: Path) -> int:
  """Deterministically export discrete ODL YAML files into a documentation YAML list.

  Args:
      odl_dir: Path to directory containing discrete ODL YAML files.
      output_yaml: Path to target documentation YAML catalog file.

  Returns:
      Number of operations exported.
  """
  catalog = _load_odl_directory(odl_dir)
  ordered_ops = [catalog[key] for key in sorted(catalog.keys())]
  output_yaml.parent.mkdir(parents=True, exist_ok=True)
  with open(output_yaml, "w", encoding="utf-8") as f:
    yaml.safe_dump(ordered_ops, f, sort_keys=True, indent=2)
  return len(ordered_ops)


def verify_docs_catalog(odl_dir: Path, docs_yaml: Path) -> bool:
  """Verify that a generated documentation YAML catalog matches the ODL source of truth.

  Args:
      odl_dir: Path to directory containing discrete ODL YAML files.
      docs_yaml: Path to documentation YAML catalog file to verify.

  Returns:
      True if ``docs_yaml`` matches the deterministic ODL catalog export, False otherwise.
  """
  if not docs_yaml.is_file():
    return False
  try:
    with open(docs_yaml, "r", encoding="utf-8") as f:
      loaded = yaml.safe_load(f)
  except (OSError, yaml.YAMLError, ValueError, RuntimeError):
    return False
  if not isinstance(loaded, list):
    return False
  catalog = _load_odl_directory(odl_dir)
  expected = [catalog[key] for key in sorted(catalog.keys())]
  return loaded == expected


def verify_roundtrip(odl_dir: Path, json_path: Path) -> bool:
  """Verify serialization roundtrip between discrete YAMLs and compiled JSON.

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

  yaml_ops = _load_odl_directory(odl_dir)
  if set(yaml_ops.keys()) != set(compiled_data.keys()):
    return False

  return all(compiled_data.get(key) == yaml_val for key, yaml_val in yaml_ops.items())


def decompile_catalog(json_path: Path, output_dir: Path) -> int:
  """Decompile a compiled JSON catalog back into discrete YAML files.

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


def _run_verify_mode(odl_dir: Path, output_json: Path, docs_yaml: Path | None) -> int:
  """Execute ``--verify-only`` checks for JSON and optional docs YAML catalogs.

  Args:
      odl_dir: Path to discrete ODL YAML directory.
      output_json: Path to compiled JSON catalog.
      docs_yaml: Optional path to generated docs YAML catalog.

  Returns:
      Exit code (0 on success, 1 on mismatch).
  """
  ok = verify_roundtrip(odl_dir, output_json)
  if ok and docs_yaml is not None:
    ok = verify_docs_catalog(odl_dir, docs_yaml)
  if ok:
    print("✅ Roundtrip verification passed.")
    return 0
  print("❌ Roundtrip verification failed.")
  return 1


def main(args: typing.Sequence[str] | None = None) -> int:
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
    "--docs-yaml",
    type=Path,
    default=None,
    help="Optional path to deterministically generated docs YAML catalog",
  )
  parser.add_argument(
    "--verify-only",
    action="store_true",
    help="Verify roundtrip equivalence without compiling",
  )
  parsed = parser.parse_args(args)

  if parsed.verify_only:
    return _run_verify_mode(parsed.odl_dir, parsed.output_json, parsed.docs_yaml)

  count = compile_catalog(parsed.odl_dir, parsed.output_json, validate=True)
  print(f"Compiled and validated {count} operations into {parsed.output_json}.")

  if parsed.docs_yaml is not None:
    export_docs_catalog(parsed.odl_dir, parsed.docs_yaml)

  ok = verify_roundtrip(parsed.odl_dir, parsed.output_json)
  if ok and parsed.docs_yaml is not None:
    ok = verify_docs_catalog(parsed.odl_dir, parsed.docs_yaml)
  if not ok:
    print("❌ Roundtrip verification failed after compilation.")
    return 1

  print("✅ Roundtrip serialization verified.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
