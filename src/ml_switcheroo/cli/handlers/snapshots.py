"""
Snapshot and Sync Command Handlers.

This module manages the "Spoke" side of the Knowledge Base (Snapshots & Overlays).
It handles:
1.  **Syncing**: Linking Abstract Standards (Hub) to installed framework implementations.
2.  **Snapshotting**: Capturing API surfaces for Ghost Mode support.
"""

import json
import importlib.metadata
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

from ml_switcheroo.semantics.paths import resolve_semantics_dir, resolve_snapshots_dir
from ml_switcheroo.discovery.syncer import FrameworkSyncer
from ml_switcheroo.frameworks.base import (
  StandardCategory,
  get_adapter,
  GhostRef,
  FrameworkAdapter,
  SNAPSHOT_DIR as DEFAULT_SNAP_DIR,
)
from ml_switcheroo.frameworks import available_frameworks
from ml_switcheroo.utils.console import (
  log_info,
  log_success,
  log_error,
  log_warning,
)


def handle_snapshot(out_dir: Optional[Path]) -> int:
  """
  Handles the 'snapshot' command.

  Scans installed frameworks and dumps their API signatures to JSON files.
  These snapshots allow the engine to operate in "Ghost Mode" (e.g., in WebAssembly)
  without requiring the heavy frameworks to be installed.

  Args:
      out_dir: Optional custom output directory. Defaults to package/snapshots.

  Returns:
      int: Exit code.
  """
  target_dir = out_dir or DEFAULT_SNAP_DIR

  log_info(f"Starting Snapshot Capture to {target_dir}")

  frameworks = available_frameworks()

  if not frameworks:
    log_error("No frameworks registered in ml-switcheroo. Check installation.")
    return 1

  processed = 0
  for fw in frameworks:
    data = _capture_framework(fw)
    if data:
      _save_snapshot(fw, data, target_dir)
      processed += 1

  if processed == 0:
    log_warning("No snapshots were generated. Are Torch/Keras/JAX installed?")
    return 1

  log_success(f"Capture complete. Generated {processed} snapshot headers.")
  return 0


def handle_sync(framework: str) -> int:
  """
  Handles the 'sync' command.

  Links the Abstract Standards (Hub) to the concrete implementation in a
  specific framework. This process involves:
  1.  Reading the Specs from `semantics/`.
  2.  Introspecting the installed framework to find matching APIs.
  3.  **Merging Static Definitions** from the Adapter (for Ghost support).
  4.  Applying Adapter-specific manual wiring (plugins, templates).
  5.  Writing the results to the Snapshot Overlay (`snapshots/{fw}_mappings.json`).

  Args:
      framework: The framework key to sync (e.g. 'torch', 'jax').

  Returns:
      int: Exit code.
  """
  # Resolve paths relative to package
  sem_dir = resolve_semantics_dir()
  snap_dir = resolve_snapshots_dir()

  # 1. Load or Initialize current Snapshot (to preserve other mappings)
  ver = _get_pkg_version(framework)
  snap_path = snap_dir / f"{framework}_v{ver}_map.json"
  snapshot_data = {"__framework__": framework, "mappings": {}}

  if snap_path.exists():
    try:
      with open(snap_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        if loaded.get("__framework__") == framework:
          snapshot_data = loaded
    except Exception as e:
      log_warning(f"Could not read existing snapshot at {snap_path}: {e}")

  # Existing entries we want to preserve/respect
  existing_mappings = snapshot_data.get("mappings", {})
  adapter = get_adapter(framework)

  # 2. Run Syncer on Spec data
  syncer = FrameworkSyncer()

  # We include 'k_discovered.json' to ensure discovered layers are synced
  tiers = [
    "k_array_api.json",
    "k_neural_net.json",
    "k_framework_extras.json",
    "k_discovered.json",
  ]

  total_found = 0

  for filename in tiers:
    spec_path = sem_dir / filename
    if not spec_path.exists():
      continue

    try:
      with open(spec_path, "r", encoding="utf-8") as f:
        tier_data = json.load(f)

      # Inject existing mappings into tier_data so Syncer can see them (and skip if needed)
      for op_name, details in tier_data.items():
        if op_name in existing_mappings:
          if "variants" not in details:
            details["variants"] = {}
          details["variants"][framework] = existing_mappings[op_name]

      # Execute Discovery (Modifies tier_data in-place)
      syncer.sync(tier_data, framework)

      # Extract results
      count_in_tier = 0
      for op_name, details in tier_data.items():
        variants = details.get("variants", {})
        if framework in variants:
          # Update snapshot mapping definition
          snapshot_data["mappings"][op_name] = variants[framework]
          count_in_tier += 1

      if count_in_tier > 0:
        log_info(f"Scanned {filename}: Found {count_in_tier} entries.")
        total_found += count_in_tier

    except Exception as e:
      log_warning(f"Error processing {filename}: {e}")

  # 3. Merge Static Definitions from Adapter (Ghost Resilience)
  # This ensures fundamental ops are mapped even if the specific library isn't installed
  if adapter and hasattr(adapter, "definitions"):
    static_defs = adapter.definitions
    if static_defs:
      log_info(f"Merging {len(static_defs)} static definitions from Adapter.")
      for op_name, map_model in static_defs.items():
        # We fill gaps. If dynamic scan found a specific version, keep it.
        # If nothing found (e.g. library missing or hard to scan), use static definition.
        if op_name not in snapshot_data["mappings"]:
          snapshot_data["mappings"][op_name] = map_model.model_dump(exclude_unset=True)
          total_found += 1

  # 4. Apply Manual Wiring (Plugins & Templates) from Adapter
  if adapter and hasattr(adapter, "apply_wiring"):
    try:
      adapter.apply_wiring(snapshot_data)
      log_info(f"Applied manual wiring rules for {framework}.")
      # Assume wiring modified the dict, treat as update found
      total_found += 1
    except Exception as e:
      log_warning(f"Wiring failed for {framework}: {e}")

  # 5. Write Snapshot
  if total_found > 0 or snap_path.exists():
    if not snap_dir.exists():
      snap_dir.mkdir(parents=True, exist_ok=True)

    with open(snap_path, "w", encoding="utf-8") as f:
      json.dump(snapshot_data, f, indent=2, sort_keys=True)

    log_success(f"Synced complete. Overlay updated at [path]{snap_path.name}[/path]")
  else:
    log_info(f"No mappings found/updated for {framework}.")

  return 0


# --- Helpers ---


def _get_pkg_version(package_name: str) -> str:
  """
  Retrieves installed package version string safely.

  Args:
      package_name: The pip package name.

  Returns:
      Version string or "unknown".
  """
  try:
    if package_name == "flax_nnx":
      package_name = "flax"
    return importlib.metadata.version(package_name)
  except Exception:
    return "unknown"


def _capture_framework(fw_name: str) -> Dict[str, Any]:
  """
  Runs API collection for a single framework.

  Delegates to the framework's Adapter to scan all standardized categories.
  Used by `handle_snapshot`.

  Args:
      fw_name: The framework identifier.

  Returns:
      Dictionary containing captured API signatures and version metadata.
  """
  adapter: FrameworkAdapter = get_adapter(fw_name)
  if not adapter:
    log_warning(f"Skipping {fw_name}: No adapter found.")
    return {}

  version = _get_pkg_version(fw_name)
  if version == "unknown":
    log_warning(f"Skipping {fw_name}: Library not installed in this environment.")
    return {}

  log_info(f"Scanning {fw_name} v{version}...")

  snapshot_data = {"version": version, "categories": {}}
  found_any = False

  # Iterate all standard categories defined in the Protocol
  for category in StandardCategory:
    try:
      # Polymorphic call to the adapter's implementation
      refs = adapter.collect_api(category)

      if refs:
        found_any = True
        snapshot_data["categories"][category.value] = [ref.model_dump(exclude_unset=True) for ref in refs]
        print(f"  - Found {len(refs)} {category.value} definitions.")
    except Exception as e:
      log_error(f"  FAILED collecting {category.value}: {e}")

  if not found_any:
    return {}

  return snapshot_data


def _save_snapshot(fw_name: str, data: Dict[str, Any], target_dir: Path):
  """
  Writes snapshot data to a JSON file.

  File naming convention: `{fw_name}_v{version}.json`
  (e.g., `torch_v2.0.1.json`).

  Args:
      fw_name: Framework name.
      data: The snapshot dictionary.
      target_dir: Directory to save the file.
  """
  if not data:
    return

  safe_ver = data["version"].replace("+", "_").replace(" ", "_")
  filename = f"{fw_name}_v{safe_ver}.json"
  out_path = target_dir / filename

  target_dir.mkdir(parents=True, exist_ok=True)

  with open(out_path, "w", encoding="utf_8") as f:
    json.dump(data, f, indent=2)

  log_success(f"Saved snapshot: [path]{out_path}[/path]")
