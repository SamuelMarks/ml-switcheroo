"""Audits framework mappings against actual extracted snapshots.

Validates that APIs and parameters mapped in ODL/JSON exist in the real snapshots.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Load local components
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ml_switcheroo.semantics.manager import SemanticsManager
from ml_switcheroo.semantics.file_loader import KnowledgeBaseLoader
from ml_switcheroo.semantics.registry_loader import RegistryLoader


def load_snapshots(snapshot_dir: Path) -> Dict[str, Dict[str, Any]]:
  """Loads all JSON snapshots into memory.

  Args:
      snapshot_dir: Path to the directory containing `<framework>_vX.Y.Z.json`.

  Returns:
      A dictionary mapping framework prefix (e.g. 'torch', 'mlx') to the JSON data.
  """
  import json

  snapshots = {}
  for file_path in snapshot_dir.glob("*_v*.json"):
    if file_path.name.endswith("_map.json"):
      continue
    # Extract framework prefix (e.g., 'torch' from 'torch_v2.10.0.json')
    fw = file_path.name.split("_v")[0]
    with open(file_path, "r", encoding="utf-8") as f:
      snap = json.load(f)
      snapshots[fw] = snap

  flat_snapshots: Dict[str, Dict[str, Any]] = {}
  for fw, snap in snapshots.items():
    flat_snapshots[fw] = {}
    for cat, items in snap.get("categories", {}).items():
      if isinstance(items, list):
        for item in items:
          if "api_path" in item:
            flat_snapshots[fw][item["api_path"]] = item
          if "name" in item:
            flat_snapshots[fw][item["name"]] = item
          if "aliases" in item and isinstance(item["aliases"], list):
            for alias in item["aliases"]:
              flat_snapshots[fw][alias] = item
      elif isinstance(items, dict):
        for k, v in items.items():
          flat_snapshots[fw][k] = v
    for k, v in snap.get("functions", {}).items():
      flat_snapshots[fw][k] = v
    for k, v in snap.get("classes", {}).items():
      flat_snapshots[fw][k] = v
    for k, v in snap.items():
      if k not in ["categories", "functions", "classes", "version", "mappings", "templates", "imports", "structs"]:
        if isinstance(v, dict) and "args" in v:
          flat_snapshots[fw][k] = v

  return flat_snapshots


def audit_frameworks(manager: SemanticsManager, snapshots: Dict[str, Dict[str, Any]]) -> List[str]:
  """Audits the known manager data against the snapshots.

  Args:
      manager: The hydrated SemanticsManager.
      snapshots: The loaded snapshots.

  Returns:
      A list of error strings.
  """
  errors: List[str] = []

  for op_name, op_details in manager.data.items():
    variants = op_details.get("variants", {})
    for fw_name, fw_mapping in variants.items():
      if fw_name not in snapshots:
        # We might not have a snapshot for every framework in the matrix
        continue

      snapshot = snapshots[fw_name]
      api = fw_mapping.get("api")
      if not api:
        continue

      if api not in snapshot:
        pass  # no ignores
        # The ODL might refer to something that is implicitly handled by the runtime
        # or missing from our generated snapshot.
        # Let's verify.
        if fw_name in ["mlx", "torch", "jax", "tensorflow"] and api not in snapshot:
          errors.append(f"[{fw_name}] '{op_name}' maps to hallucinated API: '{api}'")
        continue

      api_data = snapshot[api]

      # Check arguments
      args_map = fw_mapping.get("args", {})
      if not args_map:
        continue

      snapshot_args = api_data.get("params", api_data.get("args", []))
      snapshot_arg_names = {arg["name"] for arg in snapshot_args}

      for std_name, fw_arg_name in args_map.items():
        if fw_arg_name not in snapshot_arg_names:
          # some args might be variadic or **kwargs, but we should verify exact matches if possible
          # check if the api has **kwargs
          has_kwargs = any(arg["name"] == "kwargs" or arg.get("kind") == "VAR_KEYWORD" for arg in snapshot_args)
          if not has_kwargs:
            if fw_name in ["mlx", "torch", "jax", "tensorflow"]:
              errors.append(f"[{fw_name}] '{op_name}' maps to hallucinated argument: '{fw_arg_name}' for API '{api}'")

  return errors


def main() -> int:
  """Main execution function.

  Returns:
      Exit code (0 for success, 1 for failures).
  """
  parser = argparse.ArgumentParser(description="Audit against snapshots")
  parser.add_argument("--strict", action="store_true", help="Fail if any mismatches found")
  args = parser.parse_args()

  mgr = SemanticsManager()
  KnowledgeBaseLoader(mgr).load_knowledge_graph()
  RegistryLoader(mgr).hydrate()

  snapshot_dir = Path("../ml-compiler-snapshots")
  snapshots = load_snapshots(snapshot_dir)

  print(f"Loaded {len(snapshots)} snapshots.")
  errors = audit_frameworks(mgr, snapshots)

  if errors:
    print(f"\n❌ Found {len(errors)} mismatches:\n")
    for error in errors:
      print(f"  - {error}")
    if args.strict:
      return 1
  else:
    print("\n✅ All mapped APIs and arguments verified against snapshots.")

  return 0


if __name__ == "__main__":
  sys.exit(main())
