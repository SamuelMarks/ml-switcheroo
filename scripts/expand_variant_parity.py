"""ODL Variant Parity Expansion Tool.

Audits ODL operator definitions for missing target framework variants
(JAX, MLX, Keras, AMD RDNA, NVIDIA SASS) and expands variant coverage
by introspecting grounded snapshot APIs, existing ODL catalog definitions,
and ISA specification files (`rdna_isa.json`, `nvidia_sass_isa.json`).
"""

import argparse
import json
import sys
import typing
from pathlib import Path

from pydantic import JsonValue
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from scripts.drain_quarantine import build_variant_entry, find_snapshot_api_match  # noqa: E402
from scripts.snapshots.loader import load_snapshots_multi  # noqa: E402


def _load_odl_catalog_maps(
  semantics_dir: typing.Optional[Path] = None,
) -> typing.Tuple[typing.Dict[str, typing.Dict[str, str]], typing.Dict[str, typing.Dict[str, str]]]:
  """Builds ISA and high-level framework variant maps dynamically from `odl.json`.

  Args:
      semantics_dir: Optional path to the `src/ml_switcheroo/semantics` directory.

  Returns:
      A tuple of `(isa_map, math_map)` keyed by canonical operation name.
  """
  sem_dir = semantics_dir or (REPO_ROOT / "src" / "ml_switcheroo" / "semantics")

  isa_map: typing.Dict[str, typing.Dict[str, str]] = {}
  math_map: typing.Dict[str, typing.Dict[str, str]] = {}

  odl_json_path = sem_dir / "odl.json"
  with open(odl_json_path, "r", encoding="utf-8") as f:
    catalog: typing.Dict[str, typing.Dict[str, JsonValue]] = json.load(f)

  for op_key, op_def in catalog.items():
    raw_variants = op_def.get("variants", {})
    variants = raw_variants if isinstance(raw_variants, dict) else {}

    op_names = {op_key}
    op_field = op_def.get("operation")
    if isinstance(op_field, str) and op_field:
      op_names.add(op_field)

    isa_entry: typing.Dict[str, str] = {}
    for isa_fw in ("rdna", "nvidia_sass"):
      v_info = variants.get(isa_fw)
      if isinstance(v_info, dict):
        v_api = v_info.get("api")
        if isinstance(v_api, str) and v_api:
          isa_entry[isa_fw] = v_api

    math_entry: typing.Dict[str, str] = {}
    hl_targets = ("torch", "jax", "mlx", "keras") if "jax" in variants else ("torch", "flax_nnx", "mlx", "keras")
    for hl_fw in hl_targets:
      v_info = variants.get(hl_fw)
      if isinstance(v_info, dict):
        v_api = v_info.get("api")
        if isinstance(v_api, str) and v_api:
          math_entry[hl_fw] = v_api

    for name in op_names:
      if isa_entry:
        isa_map.setdefault(name, {}).update(isa_entry)
      if math_entry:
        math_map.setdefault(name, {}).update(math_entry)

  # Ensure Linear / Dense parity in math_map if flax_nnx is present
  for linear_alias in ("Linear", "linear", "Dense", "dense"):
    if linear_alias in math_map:
      entry = math_map[linear_alias]
      if entry.get("torch") == "torch.nn.functional.linear" and "flax_nnx" in entry:
        entry["torch"] = "torch.nn.Linear"
        entry["keras"] = "keras.layers.Dense"

  return isa_map, math_map


ISA_ALU_MACRO_MAP, CANONICAL_MATH_MAP = _load_odl_catalog_maps()


def expand_op_variants(
  odl_data: typing.Dict[str, JsonValue],
  snapshots: typing.Mapping[str, typing.Mapping[str, typing.Union[JsonValue, typing.Hashable, typing.Sized]]],
  stem_name: typing.Optional[str] = None,
) -> int:
  """Expands missing target framework variants in an ODL dictionary.

  Args:
      odl_data: The parsed ODL YAML dictionary.
      snapshots: Loaded ground-truth snapshot dictionary.
      stem_name: Optional file stem name to evaluate as an operation candidate.

  Returns:
      Number of newly added variants.
  """
  raw_variants = odl_data.setdefault("variants", {})
  if not isinstance(raw_variants, dict):
    raw_variants = {}
    odl_data["variants"] = raw_variants
  variants = raw_variants

  raw_op_name = odl_data.get("operation", "")
  op_name = str(raw_op_name) if isinstance(raw_op_name, str) else ""

  added_count = 0

  # Candidate search names: op_name and function name from torch variant
  search_names: typing.List[str] = [op_name]
  if stem_name and stem_name not in search_names:
    search_names.append(stem_name)
  torch_var = variants.get("torch")
  torch_api = torch_var.get("api") if isinstance(torch_var, dict) else None
  if isinstance(torch_api, str) and torch_api:
    fn_name = torch_api.split(".")[-1]
    if fn_name not in search_names:
      search_names.append(fn_name)

  # 1. Expand JAX, MLX, Keras from snapshots
  for fw in ("jax", "mlx", "keras"):
    if fw not in variants and fw in snapshots:
      fw_snap = snapshots[fw]
      match: typing.Optional[typing.Tuple[str, typing.Dict[str, JsonValue]]] = None
      for candidate in search_names:
        match = find_snapshot_api_match(candidate, fw_snap)
        if match:
          break

      # Fallback for JAX from paxml or flax_nnx jnp references
      if not match and fw == "jax":
        for source_fw in ("paxml", "flax_nnx"):
          src_var = variants.get(source_fw)
          src_api = src_var.get("api", "") if isinstance(src_var, dict) else ""
          if isinstance(src_api, str) and src_api.startswith("jnp."):
            jnp_fn = src_api.split(".")[-1]
            match = find_snapshot_api_match(jnp_fn, fw_snap)
            if match:
              break

      if match:
        api_path, api_data = match
        variants[fw] = build_variant_entry(api_path, api_data)
        added_count += 1

  # 2. Expand RDNA and NVIDIA SASS from catalog-derived ISA map (case-insensitive, strip trailing _)
  clean_candidates = {c.rstrip("_").lower() for c in search_names if c}
  isa_entry = None
  for k, v in ISA_ALU_MACRO_MAP.items():
    if k.lower() in clean_candidates:
      isa_entry = v
      break

  if isa_entry:
    if "rdna" not in variants and "rdna" in isa_entry:
      variants["rdna"] = {"api": isa_entry["rdna"], "args": {}}
      added_count += 1
    if "nvidia_sass" not in variants and "nvidia_sass" in isa_entry:
      variants["nvidia_sass"] = {"api": isa_entry["nvidia_sass"], "args": {}}
      added_count += 1

  # 3. Expand missing high-level frameworks from catalog-derived math map
  math_entry = None
  for k, v in CANONICAL_MATH_MAP.items():
    if k.lower() in clean_candidates:
      math_entry = v
      break

  if math_entry:
    for fw, api_call in math_entry.items():
      if fw not in variants:
        variants[fw] = {"api": api_call, "args": {}}
        added_count += 1

  return added_count


def expand_parity_across_odl(
  odl_dir: Path,
  snapshots: typing.Mapping[str, typing.Mapping[str, typing.Union[JsonValue, typing.Hashable, typing.Sized]]],
  dry_run: bool = False,
) -> typing.Tuple[int, int]:
  """Iterates through ODL definitions and expands missing variants.

  Args:
      odl_dir: Path to directory containing ODL YAML files.
      snapshots: Ground-truth snapshot dictionaries.
      dry_run: If True, calculates changes without writing to disk.

  Returns:
      A tuple of (modified_file_count, total_variants_added).
  """
  modified_files = 0
  total_added = 0

  for yaml_path in sorted(odl_dir.glob("*.yaml")):
    try:
      with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    except yaml.YAMLError:
      continue

    if not isinstance(data, dict):
      continue

    added = expand_op_variants(data, snapshots, stem_name=yaml_path.stem)
    if added > 0:
      modified_files += 1
      total_added += added
      if not dry_run:
        with open(yaml_path, "w", encoding="utf-8") as f:
          yaml.dump(data, f, sort_keys=False, indent=2)

  return modified_files, total_added


def main(args: typing.Optional[typing.Sequence[str]] = None) -> int:
  """Command-line interface for ODL variant parity expansion.

  Args:
      args: Optional command-line argument list.

  Returns:
      Exit code (0 on success).
  """
  parser = argparse.ArgumentParser(description="Expand ODL variant parity across frameworks")
  parser.add_argument(
    "--odl-dir",
    type=Path,
    default=Path("src/ml_switcheroo/semantics/odl"),
    help="Path to ODL YAML files",
  )
  parser.add_argument(
    "--snapshots-dir",
    action="append",
    default=[],
    help="Explicit snapshot directory to load (can be specified multiple times)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Compute changes without writing to disk",
  )
  parsed = parser.parse_args(args)

  snapshot_dirs = [Path(p) for p in parsed.snapshots_dir]
  snapshot_dirs.extend(
    [
      Path("../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/snapshots"),
      Path("../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/frameworks"),
      Path("../ml-ecosystem-snapshots/snapshots"),
      Path("../ml-ecosystem-snapshots"),
    ]
  )
  snapshots = load_snapshots_multi(snapshot_dirs)

  mod_files, added_variants = expand_parity_across_odl(
    odl_dir=parsed.odl_dir,
    snapshots=snapshots,
    dry_run=parsed.dry_run,
  )

  print(f"Expanded parity across {mod_files} ODL files (+{added_variants} variants).")
  return 0


if __name__ == "__main__":
  sys.exit(main())
