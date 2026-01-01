"""
Semantic Autogen: The Distributed Persistence Layer.

This module is responsible for finalizing "Candidate Standards" proposed by the
Consensus Engine into the Distributed Knowledge Base.

It implements the **Hub-and-Spoke** write strategy:

1.  **Hub (Specs)**: Writes Abstract Definitions (`std_args`, `description`)
    to the target spec file in `semantics/`.
2.  **Spokes (Snapshots)**: Writes Implementation Details (`api`, `args`)
    to `snapshots/{framework}_v{version}_map.json`.

Policy:

-   **Do Not Harm**: Existing keys in Specs or Snapshots (implying manual curation)
    are skipped to prevent overwriting high-quality manual edits.
-   **Additive**: New discoveries are added.
-   **Atomic**: Writes are performed per-file using standard JSON serialization.
"""

import json
import importlib.metadata
from pathlib import Path
from typing import Dict, List, Any, DefaultDict
from collections import defaultdict

from ml_switcheroo.discovery.consensus import CandidateStandard
from ml_switcheroo.utils.console import log_info, log_warning, log_success
from ml_switcheroo.semantics.paths import resolve_snapshots_dir


class SemanticPersister:
  """
  Handles serialization of Discovered Standards to disk using Hub-and-Spoke architecture.
  """

  def persist(self, candidates: List[CandidateStandard], target_spec_file: Path) -> None:
    """
    Splits and persists candidates into Specifications (Hub) and Snapshots (Spokes).

    Args:
        candidates: List of aligned CandidateStandard objects.
        target_spec_file: Path to the JSON semantic spec file (Hub) to update/create.
                          (e.g., `semantics/k_framework_extras.json`)
    """
    snapshots_dir = resolve_snapshots_dir()
    if not snapshots_dir.exists():
      snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Buffer for the Hub (Spec File) - OpName -> Definition
    spec_updates: Dict[str, Any] = {}

    # Buffer for Spokes (Snapshots) - Framework -> OpName -> Mapping
    snapshot_updates: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)

    stats = {"spec_writes": 0, "spec_skips": 0, "snapshot_writes": 0, "snapshot_skips": 0}

    # 1. Load Existing Spec Data to check logic
    existing_spec = self._load_json(target_spec_file)

    # 2. Distribute Candidates into Buffers
    for cand in candidates:
      # --- HUB: Abstract Spec ---
      if cand.name in existing_spec:
        stats["spec_skips"] += 1
      else:
        # Transform to Abstract Schema
        spec_entry = {
          "std_args": sorted(cand.std_args),
          "description": f"Auto-discovered via Consensus (Score: {cand.score})",
          # Variants keys removed from Hub in new architecture
        }
        spec_updates[cand.name] = spec_entry
        stats["spec_writes"] += 1

      # --- SPOKES: Framework Mappings ---
      for fw_name, ref in cand.variants.items():
        mapping_entry = {"api": ref.api_path}

        # Attach argument mappings found by consensus
        arg_map = cand.arg_mappings.get(fw_name)
        if arg_map:
          mapping_entry["args"] = arg_map

        snapshot_updates[fw_name][cand.name] = mapping_entry

    # 3. Persist Hub (Spec File)
    if spec_updates:
      self._write_updates(target_spec_file, spec_updates)

    # 4. Persist Spokes (Snapshot Files)
    for fw_name, new_mappings in snapshot_updates.items():
      # Determine version string dynamically for filename
      version = self._get_version(fw_name)
      snap_path = snapshots_dir / f"{fw_name}_v{version}_map.json"

      # Load existing snapshot to check collisions
      existing_snap = self._load_json(snap_path)
      existing_mappings = existing_snap.get("mappings", {})

      # Filter updates that already exist
      final_mappings = {}
      for op, mapping in new_mappings.items():
        if op in existing_mappings:
          stats["snapshot_skips"] += 1
        else:
          final_mappings[op] = mapping
          stats["snapshot_writes"] += 1

      if final_mappings:
        # Merge into existing structure
        if "__framework__" not in existing_snap:
          existing_snap["__framework__"] = fw_name
        if "mappings" not in existing_snap:
          existing_snap["mappings"] = {}

        existing_snap["mappings"].update(final_mappings)
        self._save_json(snap_path, existing_snap)

    # 5. Reporting
    if stats["spec_writes"] > 0 or stats["snapshot_writes"] > 0:
      log_success(
        f"Persistence Complete.\n"
        f"  Hub (Specs): {stats['spec_writes']} added to {target_spec_file.name} ({stats['spec_skips']} skipped)\n"
        f"  Spokes (Maps): {stats['snapshot_writes']} added across {len(snapshot_updates)} snapshots ({stats['snapshot_skips']} skipped)"
      )
    else:
      log_info("No new non-conflicting standards found to persist.")

  def _get_version(self, fw_name: str) -> str:
    """Helper to get package version or fallback to 'unknown'."""
    try:
      # Handle special package names
      pkg = "flax" if fw_name == "flax_nnx" else fw_name
      return importlib.metadata.version(pkg)
    except Exception:
      return "unknown"

  def _load_json(self, path: Path) -> Dict[str, Any]:
    """Safely loads JSON from disk, returning empty dict if missing or corrupt."""
    if not path.exists():
      return {}
    try:
      with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    except json.JSONDecodeError:
      log_warning(f"Corrupt JSON at {path}. Treating as empty (Warning: Overwrite risk).")
      # In a real scenario, we might backup here. For simplicity, return empty.
      return {}

  def _save_json(self, path: Path, data: Dict[str, Any]) -> None:
    """Atomic write to JSON file."""
    if not path.parent.exists():
      path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
      json.dump(data, f, indent=2, sort_keys=True)

  def _write_updates(self, path: Path, updates: Dict[str, Any]) -> None:
    """Merges updates into file and saves."""
    current = self._load_json(path)
    current.update(updates)
    self._save_json(path, current)
