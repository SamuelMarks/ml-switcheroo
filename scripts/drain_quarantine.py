"""Quarantine Drainage and Triage Tool.

Audits and triages quarantined operators against ground-truth framework snapshots,
migrating convertible operators into discrete ODL definitions and cleansing
the quarantine file to contain only non-convertible or deprecated symbols.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_against_snapshots import load_snapshots_multi  # noqa: E402
from scripts.hydrate_odl_signatures import extract_std_args_from_params  # noqa: E402


def find_snapshot_api_match(
  op_name: str,
  fw_snapshot: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any]]]:
  """Finds a matching API in a framework snapshot for a given operator name.

  Args:
      op_name: The operator identifier (e.g. 'Amax', 'AdamW', 'Celu').
      fw_snapshot: Flattened snapshot dictionary for a framework.

  Returns:
      A tuple of (api_path, api_data) if a match is found, otherwise None.
  """
  lower_op = op_name.lower()

  # 1. Exact name match
  if op_name in fw_snapshot:
    entry = fw_snapshot[op_name]
    api_path = entry.get("api_path", op_name)
    return api_path, entry

  if lower_op in fw_snapshot:
    entry = fw_snapshot[lower_op]
    api_path = entry.get("api_path", lower_op)
    return api_path, entry

  # 2. Match by suffix in api_path
  for api_path, entry in fw_snapshot.items():
    if not isinstance(entry, dict):
      continue
    name = entry.get("name", "")
    if name in (op_name, lower_op):
      return entry.get("api_path", api_path), entry
    if api_path.endswith("." + op_name) or api_path.endswith("." + lower_op):
      return entry.get("api_path", api_path), entry

  return None


def build_variant_entry(
  api_path: str,
  api_data: Dict[str, Any],
) -> Dict[str, Any]:
  """Constructs a variant dictionary with mapped required arguments.

  Args:
      api_path: The fully qualified API path.
      api_data: The snapshot metadata dictionary for the API.

  Returns:
      A variant dictionary containing api and args mappings.
  """
  params = api_data.get("params", api_data.get("args", []))
  args_map: Dict[str, str] = {}
  for p in params:
    p_name = p.get("name", "")
    if p_name in ("self", "cls"):
      continue
    is_required = p.get("default") is None and p.get("kind") not in ("VAR_POSITIONAL", "VAR_KEYWORD")
    if is_required:
      args_map[p_name] = p_name

  return {
    "api": api_path,
    "args": args_map,
  }


def drain_quarantine_data(
  quarantine_data: Dict[str, Any],
  odl_dir: Path,
  snapshots: Dict[str, Dict[str, Any]],
  dry_run: bool = False,
  framework_order: Optional[Sequence[str]] = None,
) -> Tuple[int, int, Dict[str, Any]]:
  """Triages quarantined operators, migrating matched ops to discrete ODL files.

  Args:
      quarantine_data: Raw dictionary loaded from quarantine.yaml.
      odl_dir: Path to directory containing ODL YAML files.
      snapshots: Mapping of framework identifiers to snapshot dictionaries.
      dry_run: If True, computes triage actions without writing changes to disk.
      framework_order: Ordered sequence of frameworks to evaluate for variants.

  Returns:
      A tuple of (removed_existing_count, migrated_count, cleansed_quarantine).
  """
  if framework_order is None:
    framework_order = ("torch", "jax", "mlx", "keras", "stablehlo")

  existing_odl_ops: Set[str] = {yf.stem for yf in odl_dir.glob("*.yaml")}

  cleansed_quarantine: Dict[str, Any] = {}
  removed_existing = 0
  migrated_count = 0

  for op_name, op_spec in sorted(quarantine_data.items()):
    if not isinstance(op_spec, dict):
      cleansed_quarantine[op_name] = op_spec
      continue

    # If op already exists in ODL, simply remove from quarantine
    if op_name in existing_odl_ops:
      removed_existing += 1
      continue

    # Attempt to match against ground-truth snapshots
    matched_variants: Dict[str, Dict[str, Any]] = {}
    primary_params: Optional[List[Dict[str, Any]]] = None
    primary_doc: Optional[str] = None

    for fw in framework_order:
      if fw not in snapshots:
        continue
      match = find_snapshot_api_match(op_name, snapshots[fw])
      if match:
        api_path, api_data = match
        matched_variants[fw] = build_variant_entry(api_path, api_data)
        if primary_params is None:
          primary_params = api_data.get("params", api_data.get("args", []))
          primary_doc = api_data.get("docstring")

    if matched_variants:
      # Generate new discrete ODL definition
      desc = primary_doc or op_spec.get("description") or f"Standardized definition for {op_name}."
      first_line_desc = desc.splitlines()[0].strip() if desc else f"Standardized definition for {op_name}."
      if not first_line_desc:
        first_line_desc = f"Standardized definition for {op_name}."

      std_args = extract_std_args_from_params(primary_params or [])

      odl_entry = {
        "operation": op_name,
        "description": first_line_desc,
        "std_args": std_args,
        "variants": matched_variants,
      }

      migrated_count += 1
      if not dry_run:
        target_file = odl_dir / f"{op_name}.yaml"
        with open(target_file, "w", encoding="utf-8") as f:
          yaml.dump(odl_entry, f, sort_keys=False, indent=2)
    else:
      # Non-convertible / internal symbol remains in cleansed quarantine
      cleansed_quarantine[op_name] = op_spec

  return removed_existing, migrated_count, cleansed_quarantine


def main(args: Optional[Sequence[str]] = None) -> int:
  """Command-line interface for draining quarantine.yaml.

  Args:
      args: Optional command-line argument list.

  Returns:
      Exit code (0 on success).
  """
  parser = argparse.ArgumentParser(description="Drain and triage quarantine.yaml")
  parser.add_argument(
    "--quarantine-file",
    type=Path,
    default=Path("src/ml_switcheroo/semantics/quarantine.yaml"),
    help="Path to quarantine.yaml",
  )
  parser.add_argument(
    "--odl-dir",
    type=Path,
    default=Path("src/ml_switcheroo/semantics/odl"),
    help="Path to ODL YAML files",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Compute triage without writing changes to disk",
  )
  parsed = parser.parse_args(args)

  with open(parsed.quarantine_file, "r", encoding="utf-8") as f:
    quarantine_data = yaml.safe_load(f) or {}

  snapshot_dirs = [
    Path("../ml-framework-snapshots/src/ml_framework_snapshots/snapshots"),
    Path("../ml-compiler-snapshots"),
  ]
  snapshots = load_snapshots_multi(snapshot_dirs)

  removed, migrated, cleansed = drain_quarantine_data(
    quarantine_data=quarantine_data,
    odl_dir=parsed.odl_dir,
    snapshots=snapshots,
    dry_run=parsed.dry_run,
  )

  print(f"Removed {removed} duplicate ops already present in ODL.")
  print(f"Migrated {migrated} verified ops to ODL.")
  print(f"Cleansed quarantine retains {len(cleansed)} non-convertible symbols.")

  if not parsed.dry_run:
    with open(parsed.quarantine_file, "w", encoding="utf-8") as f:
      yaml.dump(cleansed, f, sort_keys=False, indent=2)

  return 0


if __name__ == "__main__":
  sys.exit(main())
