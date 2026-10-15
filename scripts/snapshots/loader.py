"""Snapshot finding and loading utilities."""

import importlib
import importlib.resources
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _resolve_grounding_engine() -> Any:
  """Resolve GroundingEngine class from available ecosystem or fallback packages.

  Returns:
      Any: The GroundingEngine class if resolvable, otherwise None.
  """
  for mod_name in ("ml_ecosystem_snapshots.grounding.engine", "ml_framework_snapshots.grounding.engine"):
    try:
      mod = importlib.import_module(mod_name)
      return getattr(mod, "GroundingEngine")
    except ImportError:
      pass

  for repo_name in ("ml-ecosystem-snapshots", "ml-framework-snapshots"):
    try:
      parent_src = Path(__file__).resolve().parent.parent.parent / repo_name / "src"
      if parent_src.exists() and str(parent_src) not in sys.path:
        sys.path.insert(0, str(parent_src))
      pkg_name = repo_name.replace("-", "_") + ".grounding.engine"
      mod = importlib.import_module(pkg_name)
      return getattr(mod, "GroundingEngine")
    except ImportError:
      pass

  return None


GroundingEngine: Any = _resolve_grounding_engine()


def _flatten_single_framework(fw: str, snap: Any, flat_snapshots: Dict[str, Dict[str, Any]]) -> None:
  """Flattens a raw snapshot JSON structure into a fast lookup dictionary.

  Args:
      fw: The framework name key.
      snap: The raw snapshot data (dict or list of items).
      flat_snapshots: Target dictionary mapping framework names to lookup dicts.
  """
  if fw not in flat_snapshots:
    flat_snapshots[fw] = {}

  if isinstance(snap, list):
    for item in snap:
      if isinstance(item, dict) and "mnemonic" in item:
        flat_snapshots[fw][item["mnemonic"]] = item
    return

  if not isinstance(snap, dict):
    return

  for cat, items in snap.get("categories", {}).items():
    if isinstance(items, list):
      for item in items:
        if "api_path" in item:
          api_path = item["api_path"]
          flat_snapshots[fw][api_path] = item
          if api_path.startswith("jax.numpy."):
            flat_snapshots[fw]["jnp." + api_path[len("jax.numpy.") :]] = item
          elif api_path.startswith("mlx.core."):
            flat_snapshots[fw]["mx." + api_path[len("mlx.core.") :]] = item
          elif api_path.startswith("torch.nn.functional."):
            flat_snapshots[fw]["F." + api_path[len("torch.nn.functional.") :]] = item
          elif api_path.startswith("flax.nnx."):
            flat_snapshots[fw]["nnx." + api_path[len("flax.nnx.") :]] = item
        if "name" in item:
          flat_snapshots[fw][item["name"]] = item
        if "mnemonic" in item:
          flat_snapshots[fw][item["mnemonic"]] = item
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
  for item in snap.get("operations", []):
    if isinstance(item, dict):
      if "api_path" in item:
        flat_snapshots[fw][item["api_path"]] = item
      if "class_name" in item:
        flat_snapshots[fw][item["class_name"]] = item
      if "name" in item:
        flat_snapshots[fw][item["name"]] = item
      if "aliases" in item and isinstance(item["aliases"], list):
        for alias in item["aliases"]:
          flat_snapshots[fw][alias] = item
  for k, v in snap.items():
    if k not in [
      "categories",
      "functions",
      "classes",
      "operations",
      "version",
      "mappings",
      "templates",
      "imports",
      "structs",
    ]:
      if isinstance(v, dict) and (
        fw in ("nvidia_sass", "rdna")
        or "args" in v
        or any(key in v for key in ("alu", "memory", "control_flow", "tensor_core", "inputs"))
      ):
        flat_snapshots[fw][k] = v


def load_snapshots(snapshot_dir: Path) -> Dict[str, Dict[str, Any]]:
  """Loads all JSON snapshots into memory.

  Args:
      snapshot_dir: Path to the directory containing `<framework>_vX.Y.Z.json` or exhaustive ISA files.

  Returns:
      A dictionary mapping framework prefix (e.g. 'torch', 'mlx', 'nvidia_sass') to the JSON data.
  """
  flat_snapshots: dict[str, Dict[str, Any]] = {}
  candidates = (
    list(snapshot_dir.glob("*_v*.json"))
    + list(snapshot_dir.glob("*_exhaustive.json"))
    + list(snapshot_dir.glob("*_isa.json"))
  )
  for file_path in candidates:
    if file_path.name.endswith("_map.json") or "_vunknown" in file_path.name:
      continue
    if "_v" in file_path.name:
      fw = file_path.name.split("_v")[0]
    elif "sass" in file_path.name:
      fw = "nvidia_sass"
    elif "rdna" in file_path.name:
      fw = "rdna"
    else:
      fw = file_path.stem

    if fw == "amd_rdna" or "rdna" in fw:
      fw = "rdna"
    elif "sass" in fw:
      fw = "nvidia_sass"

    with open(file_path, "r", encoding="utf-8") as f:
      snap = json.load(f)
      _flatten_single_framework(fw, snap, flat_snapshots)
      if fw == "stablehlo_exhaustive":
        _flatten_single_framework("stablehlo", snap, flat_snapshots)

  return flat_snapshots


def load_snapshots_multi(snapshot_dirs: Optional[List[Path]] = None) -> Dict[str, Dict[str, Any]]:
  """Load API snapshots from multiple directories.

  Args:
      snapshot_dirs: Optional list of directory paths to search for snapshot JSON files.

  Returns:
      Dictionary mapping framework keys to flattened symbol dictionaries.
  """
  dirs_to_search: List[Path] = []
  if snapshot_dirs is not None:
    for d in snapshot_dirs:
      dirs_to_search.append(d)
      if d.name == "snapshots" and (d.parent / "frameworks").exists():
        dirs_to_search.append(d.parent / "frameworks")
  else:
    for pkg in ("ml_ecosystem_snapshots.snapshots", "ml_framework_snapshots.snapshots"):
      try:
        snap_res = importlib.resources.files(pkg)
        dirs_to_search.append(Path(str(snap_res)))
      except Exception:
        pass
    for repo_name in ("ml-ecosystem-snapshots", "ml-framework-snapshots"):
      try:
        parent_snap = Path(__file__).resolve().parent.parent.parent / repo_name / "src" / repo_name.replace("-", "_")
        if (parent_snap / "snapshots").exists():
          dirs_to_search.append(parent_snap / "snapshots")
        if (parent_snap / "frameworks").exists():
          dirs_to_search.append(parent_snap / "frameworks")
      except Exception:
        pass
    try:
      semantics_dir = Path(__file__).resolve().parent.parent / "src" / "ml_switcheroo" / "semantics"
      if semantics_dir.exists():
        dirs_to_search.append(semantics_dir)
    except Exception:
      pass

  flat_snapshots: dict[str, Dict[str, Any]] = {}
  for snapshot_dir in dirs_to_search:
    if not snapshot_dir.exists():
      continue
    candidates = (
      list(snapshot_dir.glob("*_v*.json"))
      + list(snapshot_dir.glob("*_exhaustive.json"))
      + list(snapshot_dir.glob("*_isa.json"))
    )
    for file_path in candidates:
      if file_path.name.endswith("_map.json") or "_vunknown" in file_path.name:
        continue
      if "_v" in file_path.name:
        fw = file_path.name.split("_v")[0]
      elif "sass" in file_path.name:
        fw = "nvidia_sass"
      elif "rdna" in file_path.name:
        fw = "rdna"
      else:
        fw = file_path.stem

      if fw == "amd_rdna" or "rdna" in fw:
        fw = "rdna"
      elif "sass" in fw:
        fw = "nvidia_sass"

      with open(file_path, "r", encoding="utf-8") as f:
        snap = json.load(f)
        _flatten_single_framework(fw, snap, flat_snapshots)
        if fw == "optax_shim":
          _flatten_single_framework("optax", snap, flat_snapshots)
        elif fw == "stablehlo_exhaustive":
          _flatten_single_framework("stablehlo", snap, flat_snapshots)

  return flat_snapshots


# Strict supplementary lookup tables for validated hardware ISA instructions.


def compute_snapshot_checksums(snapshot_dirs: Optional[List[Path]] = None) -> Dict[str, str]:
  """Computes SHA-256 checksums of discovered snapshot files.

  Args:
      snapshot_dirs: Optional list of directories to scan for snapshots.

  Returns:
      A dictionary mapping snapshot file names to their SHA-256 hexadecimal digests.
  """
  import hashlib

  dirs_to_search: List[Path] = []
  if snapshot_dirs is not None:
    dirs_to_search.extend(snapshot_dirs)
  else:
    dirs_to_search.extend(
      [
        Path("../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/snapshots"),
        Path("../ml-ecosystem-snapshots/src/ml_ecosystem_snapshots/frameworks"),
        Path("../ml-framework-snapshots/src/ml_framework_snapshots/snapshots"),
        Path("../ml-framework-snapshots/src/ml_framework_snapshots/frameworks"),
        Path("src/ml_switcheroo/semantics"),
      ]
    )

  checksums: Dict[str, str] = {}
  for d in dirs_to_search:
    if not d.exists():
      continue
    candidates = sorted(set(list(d.glob("*_v*.json")) + list(d.glob("*_exhaustive.json")) + list(d.glob("*_isa.json"))))
    for file_path in candidates:
      if file_path.name.endswith("_map.json") or "_vunknown" in file_path.name:
        continue
      try:
        data = file_path.read_bytes()
        if len(data) > 0:
          checksums[file_path.name] = hashlib.sha256(data).hexdigest()
      except OSError as exc:
        import logging

        logging.getLogger(__name__).warning("Failed to read snapshot %s: %s", file_path, exc)
  return checksums
