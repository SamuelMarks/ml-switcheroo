"""ODL Signature Hydrator.

Extracts signature metadata (parameters, kinds, default values, variadics)
from ground-truth framework snapshots and hydrates `std_args` across ODL definitions.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_against_snapshots import load_snapshots_multi  # noqa: E402


def clean_variant_args(
  args_map: Dict[str, Any],
  snapshot_params: List[Dict[str, Any]],
) -> Dict[str, Any]:
  """Cleanses variant argument mappings by removing receiver and composite artifacts.

  Args:
      args_map: Dictionary of standard-to-framework argument mappings.
      snapshot_params: List of parameter dictionaries from snapshots.

  Returns:
      Cleaned argument mapping dictionary.
  """
  valid_target_names: Set[str] = set()
  for p in snapshot_params:
    if not isinstance(p, dict):
      continue
    name = str(p.get("name", ""))
    if "," in name:
      for tok in [t.strip() for t in name.split(",") if t.strip()]:
        if tok.isidentifier():
          valid_target_names.add(tok)
    elif name.isidentifier():
      valid_target_names.add(name)

  cleaned: Dict[str, Any] = {}
  for std_name, fw_name in args_map.items():
    if fw_name in ("self", "cls"):
      continue
    if isinstance(std_name, str) and "," in std_name:
      continue
    if isinstance(fw_name, str) and "," in fw_name:
      continue
    cleaned[std_name] = fw_name
  return cleaned


def simplify_type(annotation: Optional[str]) -> str:
  """Simplifies or normalizes type annotations extracted from snapshots.

  Args:
      annotation: The raw type annotation string or None.

  Returns:
      A normalized type string, e.g. 'Tensor', 'int', 'float', 'bool', or 'Any'.
  """
  if not annotation:
    return "Any"

  cleaned = annotation.strip().strip("'\"")
  if "Tensor" in cleaned or "Array" in cleaned or "ndarray" in cleaned:
    return "Tensor"
  if cleaned in ("int", "float", "bool", "str", "tuple", "list", "dict"):
    return cleaned
  return "Any"


def parse_kind(raw_kind: Optional[str]) -> tuple[str, bool]:
  """Parses a raw ParameterKind string into (kind_str, is_variadic).

  Args:
      raw_kind: The string representation of parameter kind.

  Returns:
      A tuple containing normalized kind name and whether it is variadic.
  """
  if not raw_kind:
    return "positional_or_keyword", False

  kind_upper = raw_kind.upper()
  if "VAR_POSITIONAL" in kind_upper:
    return "positional_or_keyword", True
  if "VAR_KEYWORD" in kind_upper:
    return "keyword_only", True
  if "KEYWORD_ONLY" in kind_upper:
    return "keyword_only", False
  if "POSITIONAL_ONLY" in kind_upper:
    return "positional_only", False

  return "positional_or_keyword", False


def extract_std_args_from_params(
  params: List[Dict[str, Any]],
  arg_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
  """Converts snapshot parameter definitions to ODL std_args structures.

  Args:
      params: List of parameter dictionaries from snapshots.
      arg_map: Optional target-to-standard parameter reverse mapping.

  Returns:
      List of standard argument dictionaries.
  """
  rev_map: Dict[str, str] = {}
  if arg_map:
    for std_name, fw_name in arg_map.items():
      rev_map[fw_name] = std_name

  std_args: List[Dict[str, Any]] = []
  for p in params:
    param_name = p.get("name", "")
    if param_name in ("self", "cls"):
      continue

    std_name = rev_map.get(param_name, param_name)
    kind_str, is_variadic = parse_kind(p.get("kind"))
    type_str = simplify_type(p.get("annotation"))

    arg_dict: Dict[str, Any] = {
      "name": std_name,
      "kind": kind_str,
      "is_variadic": is_variadic,
      "type": type_str,
    }

    raw_default = p.get("default")
    if raw_default is not None and raw_default not in ("None", "inspect._empty", ""):
      arg_dict["default"] = raw_default

    std_args.append(arg_dict)

  return std_args


def hydrate_odl_from_snapshots(
  odl_dir: Path,
  snapshots: Dict[str, Dict[str, Any]],
  dry_run: bool = False,
  framework_priority: Optional[Sequence[str]] = None,
  fix_args: bool = False,
) -> int:
  """Hydrates missing or string std_args in ODL files from snapshot metadata.

  Args:
      odl_dir: Path to directory containing ODL YAML files.
      snapshots: Mapping of framework names to flattened snapshot dictionaries.
      dry_run: If True, computes changes without writing back to disk.
      framework_priority: Sequence of frameworks to prioritize when extracting signatures.
      fix_args: If True, cleanses variant arguments to eliminate receiver and composite artifacts.

  Returns:
      The count of ODL files hydrated or modified.
  """
  if framework_priority is None:
    framework_priority = ("torch", "jax", "mlx", "keras", "stablehlo")

  hydrated_count = 0

  for yaml_path in sorted(odl_dir.glob("*.yaml")):
    try:
      with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    except Exception:
      continue

    if not isinstance(data, dict):
      continue

    modified = False
    variants = data.get("variants", {})

    if fix_args and isinstance(variants, dict):
      for fw, var_info in variants.items():
        if isinstance(var_info, dict) and "api" in var_info and "args" in var_info:
          api = var_info["api"]
          if fw in snapshots and api in snapshots[fw]:
            api_info = snapshots[fw][api]
            params = api_info.get("params", api_info.get("args", []))
            if params and isinstance(params, list):
              cleaned_args = clean_variant_args(var_info["args"], params)
              if cleaned_args != var_info["args"]:
                var_info["args"] = cleaned_args
                modified = True

    existing_std_args = data.get("std_args", [])
    has_rich_args = bool(existing_std_args) and all(
      isinstance(a, dict) and "name" in a and "type" in a for a in existing_std_args
    )
    if not has_rich_args and isinstance(variants, dict):
      extracted_args: Optional[List[Dict[str, Any]]] = None
      for fw in framework_priority:
        if fw in variants and isinstance(variants[fw], dict):
          api = variants[fw].get("api")
          if api and fw in snapshots and api in snapshots[fw]:
            api_info = snapshots[fw][api]
            params = api_info.get("params", api_info.get("args", []))
            if params and isinstance(params, list):
              arg_map = variants[fw].get("args", {})
              extracted_args = extract_std_args_from_params(params, arg_map)
              break

      if extracted_args:
        data["std_args"] = extracted_args
        modified = True

    if modified:
      hydrated_count += 1
      if not dry_run:
        with open(yaml_path, "w", encoding="utf-8") as f:
          yaml.dump(data, f, sort_keys=False, indent=2)

  return hydrated_count


def main(args: Optional[Sequence[str]] = None) -> int:
  """Command-line interface for ODL signature hydration.

  Args:
      args: Optional command-line argument list.

  Returns:
      Exit code (0 on success).
  """
  parser = argparse.ArgumentParser(description="Hydrate ODL std_args from snapshots")
  parser.add_argument(
    "--odl-dir",
    type=Path,
    default=Path("src/ml_switcheroo/semantics/odl"),
    help="Path to ODL YAML files",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Do not write changes to disk",
  )
  parser.add_argument(
    "--fix-args",
    action="store_true",
    help="Cleanse and align variant arguments with snapshots",
  )
  parsed = parser.parse_args(args)

  snapshots = load_snapshots_multi()

  count = hydrate_odl_from_snapshots(
    odl_dir=parsed.odl_dir,
    snapshots=snapshots,
    dry_run=parsed.dry_run,
    fix_args=parsed.fix_args,
  )

  print(f"Hydrated {count} ODL definitions from snapshots.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
